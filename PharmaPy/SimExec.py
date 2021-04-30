# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:44:44 2020

@author: dcasasor
"""

import numpy as np
import pandas as pd
from PharmaPy.ThermoModule import ThermoPhysicalManager
from PharmaPy.ParamEstim import ParameterEstimation, MultipleCurveResolution
from PharmaPy.StatsModule import StatisticsClass
from collections import OrderedDict
import time
from PharmaPy.Connections import Graph


class SimulationExec:
    def __init__(self, pure_path):

        # Interfaces
        self.ThermoInstance = ThermoPhysicalManager(pure_path)
        self.NamesSpecies = self.ThermoInstance.name_species

        # Outputs
        # self.NumSpecies = len(self.NamesSpecies)
        # self.NumStreams = len(self.Streams)
        self.StreamTable = None
        self.UnitOperations = OrderedDict()
        self.UOCounter = 0
        self.RxnSets = {}

        self.uos_instances = {}
        self.oper_mode = []
        self.connection_instances = []
        self.connection_names = []

    def LoadUOs(self):
        uos_modules = ('Reactors', 'Crystallizers', 'Containers',
                       'Evaporators', 'SolidLiquidSep')

        modules_ids = ['PharmaPy.' + elem for elem in uos_modules]
        for key, value in self.__dict__.items():
            type_val = getattr(value, '__module__', None)
            if type_val in modules_ids:
                value.name_species = self.NamesSpecies
                self.uos_instances[key] = value
                # self.oper_mode.append(value.oper_mode)

    def LoadConnections(self):
        for key, value in self.__dict__.items():
            module_name = getattr(value, '__module__', None)
            if module_name == 'PharmaPy.Connections':
                value.ThermoInstance = self.ThermoInstance
                self.connection_instances.append(value)
                self.connection_names.append(key)

    def SetConnectivity(self, incidence_matrix=None):
        self.LoadUOs()
        self.LoadConnections()

    def SolveFlowsheet(self, kwargs_run=None, pick_units=None,
                       run_subset=None):

        self.SetConnectivity()

        # Pick specific units, if given
        if pick_units is None:
            picked = self.uos_instances
        else:
            picked = pick_units

        if kwargs_run is None:
            if pick_units is None:
                keys = self.uos_instances.keys()
            else:
                keys = pick_units.keys()

            kwargs_run = {key: {} for key in keys}

        # isbatch = [elem == 'Batch' for elem in self.oper_mode]

        # Create graph and define execution order
        if len(self.uos_instances) == 1:
            execution_order = list(self.uos_instances.values())
        else:
            graph = Graph(self.connection_instances)
            execution_order = graph.topologicalSort()

        uos = self.uos_instances

        execution_order = [x for x in execution_order if x is not None]
        if run_subset is not None:
            execution_order = [execution_order[i] for i in run_subset]

        # Run loop
        for instance in execution_order:
            uo_id = list(uos.keys())[list(uos.values()).index(instance)]

            print()
            print('{}'.format('-'*30))
            print('Running {}'.format(uo_id))
            print('{}'.format('-'*30))
            print()

            for conn in self.connection_instances:
                if conn.destination_uo is instance:
                    conn.TransferData()

            instance.solve_unit(**kwargs_run.get(uo_id, {}))

            uo_type = instance.__module__
            if uo_type != 'PharmaPy.Containers':
                instance.flatten_states()

            print()
            print('Done!')
            print()

            # Connectivity
            for conn in self.connection_instances:
                if conn.source_uo is instance:
                    conn.ReceiveData()  # receive phases from upstream uo

    def GetStreamTable(self, basis='mass'):

        if basis == 'mass':
            fields_phase = ['temp', 'pres', 'mass', 'vol', 'mass_frac']
            fields_stream = ['mass_flow', 'vol_flow']

            frac_preffix = 'w_{}'
        elif basis == 'mole':
            fields_phase = ['temp', 'pres', 'moles', 'vol', 'mole_frac']
            fields_stream = ['mole_flow', 'vol_flow']

            frac_preffix = 'x_{}'

        stream_cont = []
        index_stream = []
        index_phase = []
        for ind, stream in enumerate(self.connection_instances):
            matter_obj = stream.Matter

            if matter_obj.__module__ == 'PharmaPy.MixedPhases':
                phase_list = matter_obj.Phases  # TODO: change this
            else:
                phase_list = [matter_obj]

            for phase in phase_list:
                index_stream.append(self.connection_names[ind])
                index_phase.append(phase.__class__.__name__)
                stream_info = []
                for field in fields_phase:
                    value_phase = getattr(phase, field, None)
                    stream_info.append(np.atleast_1d(value_phase))

                for field in fields_stream:
                    value_stream = getattr(phase, field, None)
                    stream_info.append(np.atleast_1d(value_stream))

                stream_info = np.concatenate(stream_info)
                stream_cont.append(stream_info)

        cols = fields_phase[:-1] + \
            [frac_preffix.format(ind) for ind in self.NamesSpecies] + \
            fields_stream

        indexes = zip(*(index_stream, index_phase))
        idx = pd.MultiIndex.from_tuples(indexes, names=('Stream', 'Phase'))
        stream_table = pd.DataFrame(stream_cont, index=idx, columns=cols)

        cols_reorder = fields_phase[:-1] + fields_stream + \
            [frac_preffix.format(ind) for ind in self.NamesSpecies]

        stream_table = stream_table[cols_reorder]
        # stream_table[stream_table == 0] = None

        return stream_table

    def SetParamEstimation(self, x_data,
                           param_seed=None, y_data=None, spectra=None,
                           fit_spectra=False,
                           phase_modifiers=None,
                           measured_ind=None, optimize_flags=None,
                           df_dtheta=None, df_dy=None,
                           covar_data=None,
                           pick_unit=None):

        self.LoadUOs()

        if len(self.uos_instances) == 1:
            target_unit = list(self.uos_instances.values())[0]
            # target_unit.reset_states = True
        else:
            if pick_unit is None:
                raise RuntimeError("Two or more unit operations detected. "
                                   "Select one using the 'pick_unit' argument")
            else:
                pass  # remember setting reset_states to True!!

        if phase_modifiers is not None:
            phase_modifiers = [(modifier, ) for modifier in phase_modifiers]

        # Get 1D array of parameters from the UO class
        if param_seed is not None:
            target_unit.Kinetics.set_params(param_seed)

        param_seed = target_unit.Kinetics.concat_params()
        param_seed = param_seed[target_unit.mask_params]

        name_params = []

        for ind, logic in enumerate(target_unit.mask_params):
            if logic:
                name_params.append(target_unit.Kinetics.name_params[ind])

        name_states = target_unit.states_uo

        # Instantiate parameter estimation
        if fit_spectra:
            self.ParamInst = MultipleCurveResolution(
                target_unit.paramest_wrapper,
                param_seed, x_data, spectra,
                args_fun=phase_modifiers, measured_ind=measured_ind,
                optimize_flags=optimize_flags,
                df_dtheta=df_dtheta, df_dy=df_dy, covar_data=covar_data,
                name_params=name_params, name_states=name_states)
        else:
            self.ParamInst = ParameterEstimation(
                target_unit.paramest_wrapper,
                param_seed, x_data, y_data,
                args_fun=phase_modifiers, measured_ind=measured_ind,
                optimize_flags=optimize_flags,
                df_dtheta=df_dtheta, df_dy=df_dy, covar_data=covar_data,
                name_params=name_params, name_states=name_states)

    def EstimateParams(self, optim_options=None, method='LM', bounds=None,
                       verbose=True):
        tic = time.time()
        results = self.ParamInst.optimize_fn(optim_options=optim_options,
                                             method=method,
                                             bounds=bounds, verbose=verbose)
        toc = time.time()

        elapsed = toc - tic

        print('Optimization time: {:.2e} s.'.format(elapsed))

        return results

    def CreateStatsObject(self, alpha=0.95):
        statInst = StatisticsClass(self.ParamInst, alpha=alpha)
        return statInst
