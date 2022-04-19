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
from PharmaPy.Connections import Graph, get_inputs

from PharmaPy.Commons import trapezoidal_rule, check_steady_state


class SimulationExec:
    def __init__(self, pure_path, time_limits={}):

        # Interfaces
        self.ThermoInstance = ThermoPhysicalManager(pure_path)
        self.NamesSpecies = self.ThermoInstance.name_species

        # Outputs
        # self.NumSpecies = len(self.NamesSpecies)
        # self.NumStreams = len(self.Streams)
        self.StreamTable = None
        self.UnitOperations = OrderedDict()
        self.UOCounter = 0

        self.uos_instances = {}
        self.oper_mode = []
        self.connection_instances = []
        self.connection_names = []

        self.time_limits = time_limits

    def LoadUOs(self):
        uos_modules = ('Reactors', 'Crystallizers', 'Containers',
                       'Evaporators', 'SolidLiquidSep', 'Drying_Model')

        modules_ids = ['PharmaPy.' + elem for elem in uos_modules]
        for key, value in self.__dict__.items():
            type_val = getattr(value, '__module__', None)
            if type_val in modules_ids:
                value.id_uo = key
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

                value.name = key

    def _get_ordered_uos_names(self, execution_order, uos_dict):
        names = []
        for obj in execution_order:
            for key, uo in uos_dict.items():
                if uo is obj:
                    names.append(key)

        return names

    def SolveFlowsheet(self, kwargs_run=None, pick_units=None, verbose=True,
                       uos_steady_state=None, tolerances_ss=None, ss_time=0,
                       kwargs_ss=None):

        if len(self.uos_instances) == 0:
            self.LoadUOs()
            self.LoadConnections()

        # Pick specific units, if given
        if kwargs_run is None:
            if pick_units is None:
                keys = self.uos_instances.keys()
            else:
                keys = pick_units

            kwargs_run = {key: {} for key in keys}

        if kwargs_ss is None:
            kwargs_ss = {}

        # isbatch = [elem == 'Batch' for elem in self.oper_mode]

        # Create graph and define execution order
        if len(self.uos_instances) == 1:
            execution_order = list(self.uos_instances.values())
        else:
            graph = Graph(self.connection_instances)
            execution_order = graph.topologicalSort()

        uos = self.uos_instances

        execution_order = [x for x in execution_order if x is not None]
        execution_names = self._get_ordered_uos_names(execution_order,
                                                      self.uos_instances)

        if pick_units is not None:
            ordered_names = []
            ordered_uos = []
            for name in pick_units:
                if name in execution_names:
                    ordered_names.append(name)
                    ind = execution_names.index(name)
                    ordered_uos.append(execution_order[ind])

            execution_order = ordered_uos
            execution_names = ordered_names

        if tolerances_ss is None:
            tolerances_ss = {}

        # Run loop
        for ind, instance in enumerate(execution_order):
            uo_id = list(uos.keys())[list(uos.values()).index(instance)]

            if verbose:
                print()
                print('{}'.format('-'*30))
                print('Running {}'.format(uo_id))
                print('{}'.format('-'*30))
                print()

            for conn in self.connection_instances:
                if conn.destination_uo is instance:
                    conn.ReceiveData()  # receive phases from upstream uo
                    conn.TransferData()

            kwargs_uo = kwargs_run.get(uo_id, {})

            tau = 0
            if hasattr(instance, '_get_tau'):
                tau = instance._get_tau()

            ss_time += tau

            if uos_steady_state is not None:
                if execution_names[ind] in uos_steady_state:
                    if instance.__class__.__name__ == 'Mixer':
                        pass
                    else:
                        tolerances = tolerances_ss.get(execution_names[ind],
                                                       1e-6)

                        kw_ss = kwargs_ss.get(execution_names[ind], None)

                        if kw_ss is None:
                            kw_ss = {'tau': tau, 'time_stop': ss_time,
                                     'threshold': tolerances}

                        else:
                            # TODO: should we keep this?
                            kw_ss['threshold'] = tolerances

                            if 'tau' not in kw_ss.keys():
                                kw_ss['tau'] = tau

                        ss_event = {'callable': check_steady_state,
                                    'num_conditions': 1,
                                    'event_name': 'steady state',
                                    'kwargs': kw_ss
                                    }

                        instance.state_event_list = [ss_event]
                        kwargs_uo['any_event'] = False

            instance.solve_unit(**kwargs_uo)

            uo_type = instance.__module__
            if uo_type != 'PharmaPy.Containers':
                instance.flatten_states()

            if verbose:
                print()
                print('Done!')
                print()

            # # Connectivity
            # for conn in self.connection_instances:
            #     if conn.source_uo is instance:
            #         conn.ReceiveData()  # receive phases from upstream uo
            #         conn.TransferData()

        self.execution_order = execution_order

        time_processing = np.zeros(len(execution_order))
        for ind, uo in enumerate(execution_order):
            if hasattr(uo, 'timeProf'):
                time_processing[ind] = uo.timeProf[-1] - uo.timeProf[0]

        self.time_processing = time_processing


    def GetStreamTable(self, basis='mass'):

        # # TODO: include inlets and holdups in the stream table
        # inlets, holdups, _, inlets_id, holdups_id = self.get_raw_objects()

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

            if matter_obj is None:
                index_stream.append(self.connection_names[ind])
                index_phase.append(None)
                stream_info = [np.nan] * (len(fields_phase) +
                                          len(fields_stream))

                stream_cont.append(stream_info)
            else:

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

    def SetParamEstimation(self, x_data, y_data=None, param_seed=None,
                           wrapper_kwargs=None,
                           spectra=None,
                           fit_spectra=False, global_analysis=True,
                           phase_modifiers=None, control_modifiers=None,
                           measured_ind=None, optimize_flags=None,
                           jac_fun=None,
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

        if phase_modifiers is None:
            if control_modifiers is None:
                phase_modifiers = [phase_modifiers]
            else:
                phase_modifiers = [phase_modifiers] * len(control_modifiers)

        if control_modifiers is None:
            if phase_modifiers is None:
                control_modifiers = [control_modifiers]
            else:
                control_modifiers = [control_modifiers] * len(phase_modifiers)

        if wrapper_kwargs is None:
            wrapper_kwargs = {}

        keys = ['modify_phase', 'modify_controls']

        kwargs_wrapper = list(zip(phase_modifiers, control_modifiers))

        kwargs_wrapper = [dict(zip(keys, item)) for item in kwargs_wrapper]

        for di in kwargs_wrapper:
            di.update({'run_args': wrapper_kwargs})

        # if len(kwargs_wrapper) == 1:
        #     kwargs_wrapper = kwargs_wrapper[0]

        # Get 1D array of parameters from the UO class
        if param_seed is not None:
            target_unit.Kinetics.set_params(param_seed)

        if hasattr(target_unit, 'Kinetics'):
            param_seed = target_unit.Kinetics.concat_params()
        else:
            param_seed = target_unit.params
        # param_seed = param_seed[target_unit.mask_params]

        name_params = []

        for ind, logic in enumerate(target_unit.mask_params):
            if logic:
                if hasattr(target_unit, 'Kinetics'):
                    name_params.append(target_unit.Kinetics.name_params[ind])
                else:
                    name_params.append(target_unit.name_params[ind])

        name_states = target_unit.states_uo

        # Instantiate parameter estimation
        if fit_spectra:
            self.ParamInst = MultipleCurveResolution(
                target_unit.paramest_wrapper,
                param_seed, x_data, spectra, global_analysis,
                kwargs_fun=kwargs_wrapper, measured_ind=measured_ind,
                optimize_flags=optimize_flags,
                jac_fun=jac_fun, covar_data=covar_data,
                name_params=name_params, name_states=name_states)
        else:
            self.ParamInst = ParameterEstimation(
                target_unit.paramest_wrapper,
                param_seed, x_data, y_data,
                kwargs_fun=kwargs_wrapper, measured_ind=measured_ind,
                optimize_flags=optimize_flags,
                jac_fun=jac_fun, covar_data=covar_data,
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

    def get_equipment_size(self):
        size_equipment = {}

        for key, instance in self.uos_instances.items():
            if hasattr(instance, 'vol_tot'):
                size_equipment[key] = instance.vol_tot
            elif hasattr(instance, 'vol_phase'):
                off_vol = instance.vol_offset
                size_equipment[key] = instance.vol_phase / off_vol

            elif hasattr(instance, 'area_filt'):
                size_equipment[key] = instance.area_filt

        return size_equipment

    def GetCAPEX(self, size_equipment=None, k_vals=None, b_vals=None,
                 cepci_vals=None, f_pres=None, f_mat=None, min_capacity=None):

        if size_equipment is None:
            size_equipment = self.get_equipment_size()

        num_equip = len(size_equipment)
        name_equip = size_equipment.keys()
        if cepci_vals is None:
            cepci_vals = np.ones(2)

        if f_pres is None:
            f_pres = np.ones(num_equip)

        if f_mat is None:
            f_mat = np.ones(num_equip)

        if k_vals is None:
            return size_equipment
        else:
            capacities = np.array(list(size_equipment.values()))

            if min_capacity is None:
                a_corr = capacities
            else:
                a_corr = np.maximum(min_capacity, capacities)

            k1, k2, k3 = k_vals.T
            cost_zero = 10**(k1 + k2*np.log10(a_corr) + k3*np.log10(a_corr)**2)

            b1, b2 = b_vals.T

            f_bare = b1 + b2 * f_mat * f_pres
            cost_equip = cost_zero * f_bare

            scale_corr = np.ones_like(capacities)
            if min_capacity is not None:
                for ind, capac in enumerate(capacities):
                    if capac < min_capacity[ind]:
                        scale_corr[ind] = (capac / min_capacity[ind])**0.6

            cost_equip *= scale_corr

            cost_equip = dict(zip(name_equip, cost_equip))

            return cost_equip

    def GetLabor(self, wage=35, num_weeks=48):
        # TODO: per/hour (per/shift) cost?
        has_solids = []
        is_batch = []
        uo_names = []

        for key, uo in self.uos_instances.items():
            if uo.__class__.__name__ != 'Mixer':

                if hasattr(uo, 'Phases'):
                    if isinstance(uo.Phases, (list, tuple)):
                        is_solid = [phase.__class__.__name__ == 'SolidPhase'
                                    for phase in uo.Phases]
                    else:
                        is_solid = [
                            uo.Phases.__class__.__name__ == 'SolidPhase']
                else:
                    is_solid = [False]  # Mixers

                has_solids.append(any(is_solid))

                oper = uo.oper_mode == 'Batch' or uo.oper_mode == 'Semibatch'
                is_batch.append(oper)
                uo_names.append(key)

        has_solids = np.array(has_solids, dtype=bool)
        is_batch = np.array(is_batch, dtype=bool)

        # Number of operators per shift
        num_workers = has_solids * (2 + is_batch) + ~has_solids * (1 + is_batch)

        hr_week = 40
        labor_cost = 1.20 * num_workers * 5 * (hr_week * num_weeks) * wage  # USD/yr

        labor_array = np.column_stack(
            (has_solids, is_batch, num_workers, labor_cost))

        labor_df = pd.DataFrame(labor_array, index=uo_names,
                                columns=('has_solids', 'is_batch',
                                         'num_workers', 'labor_cost'))
        return labor_df

    def get_raw_objects(self):
        raw_inlets = []
        inlets_ids = []

        raw_holdups = []
        holdups_ids = []

        time_inlets = []

        for uo in self.uos_instances.values():
            # Inlets (flows)
            if hasattr(uo, 'Inlet'):
                if uo.Inlet is not None:
                    if uo.Inlet.y_upstream is None:
                        inlet = getattr(uo, 'Inlet_orig', getattr(uo, 'Inlet'))
                        raw_inlets.append(inlet)

                        if uo.oper_mode == 'Batch':
                            time_inlets.append(1)
                        elif uo.Inlet.DynamicInlet is not None:
                            time_inlets.append(uo.timeProf)
                        else:
                            elapsed = uo.timeProf[-1] - uo.timeProf[0]
                            time_inlets.append(elapsed)

                        inlets_ids.append(uo.id_uo)

            elif uo.__class__.__name__ == 'Mixer':
            # hasattr(uo, 'Inlets'):
                for inlet in uo.Inlets:
                    if inlet.y_upstream is None:
                        raw_inlets.append(inlet)

                        if uo.oper_mode == 'Batch':
                            time_inlets.append(1)
                        elif inlet.DynamicInlet is not None:
                            time_inlets.append(uo.timeProf)
                        else:
                            elapsed = uo.timeProf[-1] - uo.timeProf[0]
                            time_inlets.append(elapsed)

                        inlets_ids.append(uo.id_uo)

            # Initial holdups
            if hasattr(uo, '__original_phase__'):
                if uo.oper_mode == 'Continuous':
                    raw_holdups.append(uo.__original_phase__)
                    holdups_ids.append(uo.id_uo)
                elif not uo.material_from_upstream:
                    raw_holdups.append(uo.__original_phase__)
                    holdups_ids.append(uo.id_uo)

        return (raw_inlets, raw_holdups, time_inlets, inlets_ids,
                holdups_ids)

    def GetRawMaterials(self, include_holdups=True, steady_state=False):
        name_equip = self.uos_instances.keys()

        # Raw materials
        (raw_flows, raw_holdups, time_flows,
         flow_idx, holdup_idx) = self.get_raw_objects()

        # Flows
        fracs = []
        masses_inlets = np.zeros(len(raw_flows))
        for ind, obj in enumerate(raw_flows):
            if hasattr(obj, 'DynamicInlet') and obj.DynamicInlet is not None:
                # qty_str = ('mass_flow', 'mole_flow')
                inputs = obj.evaluate_inputs(time_flows[ind])

                if 'mole_flow' in inputs:
                    flow_profile = inputs['mole_flow'] * obj.mw_av / 1000
                else:
                    flow_profile = inputs['mass_flow']
                # for string in qty_str:
                #     massprof = mass_profile[string]
                #     isarray = isinstance(massprof, np.ndarray)

                #     if isarray:
                #         qty_unit = string
                #         mass_profile = massprof
                #         break
                # if qty_unit == 'mole_flow':

                if steady_state:
                    masses_inlets[ind] = flow_profile[-1] * \
                        (time_flows[ind][-1] - time_flows[ind][0])
                else:
                    masses_inlets[ind] = trapezoidal_rule(time_flows[ind],
                                                          flow_profile)

                # pass

            else:
                try:
                    masses_inlets[ind] = obj.mass_flow * time_flows[ind]
                except:
                    masses_inlets[ind] = obj.mass

                # masses_inlets[ind] *= time_flows

            fracs.append(obj.mass_frac)

        fracs = np.array(fracs)

        inlet_comp_mass = (fracs.T * masses_inlets).T

        holdup_comp_mass = []
        if include_holdups:
            fracs = []
            masses_holdups = np.zeros(len(raw_holdups))
            for ind, obj in enumerate(raw_holdups):
                if isinstance(obj, list):
                    masa = [elem.mass for elem in obj]
                    masa = np.array(masa)

                    frac = [elem.mass_frac for elem in obj]
                    frac = (np.array(frac).T * masa).T
                    frac = frac.sum(axis=0)

                    mass = 1
                else:
                    mass = obj.mass
                    frac = obj.mass_frac

                masses_holdups[ind] = mass
                fracs.append(frac)

            fracs = np.array(fracs)

            holdup_comp_mass = (fracs.T * masses_holdups).T

        if len(holdup_comp_mass) == 0:
            holdup_comp_mass = np.zeros((len(name_equip),
                                         len(self.NamesSpecies))
                                        )

            holdup_idx = name_equip

        if len(inlet_comp_mass) == 0:
            inlet_comp_mass = np.zeros((len(name_equip),
                                        len(self.NamesSpecies))
                                       )

            flow_idx = name_equip

        flow_df = pd.DataFrame(inlet_comp_mass, index=flow_idx,
                               columns=self.NamesSpecies)

        holdup_df = pd.DataFrame(holdup_comp_mass, index=holdup_idx,
                                 columns=self.NamesSpecies)

        raw_df = pd.concat((flow_df, holdup_df), axis=0,
                           keys=('inlets', 'holdups'))
        return raw_df

    def GetDuties(self, full_output=False):
        """
        Get heat duties for all equipment that calculates an energy balance.

        Parameters
        ----------
        full_output : bool, optional
            if True, duties and duty types are returened. The default is False.

        Returns
        -------
        heat_duties : pandas dataframe
            heat duties [J].

        duties_ids : numpy array
            2D array with first column containing heating type and
            second column containing refrigeration type, according to the
            following convention:

            refrigeration: -2, -1, 0 (0 corresponding to cooling water)
            heating: 1, 2, 3 (1 corresponding to low pressure steam)

        """
        heat_duties = []
        equipment_ids = []
        duty_ids = []

        for key, instance in self.uos_instances.items():
            if hasattr(instance, 'heat_duty'):
                duty_ids.append(instance.duty_type)

                heat_duties.append(instance.heat_duty)
                equipment_ids.append(key)

        heat_duties = np.array(heat_duties)
        heat_duties = pd.DataFrame(heat_duties, index=equipment_ids,
                                   columns=['heating', 'cooling'])

        duties_ids = np.array(duty_ids)

        if full_output:
            return heat_duties, duties_ids
        else:
            return heat_duties

    def GetOPEX(self, cost_raw, include_holdups=True, steady_raw=False,
                lumped=False, kwargs_items=None):

        opex_items = ('duties', 'raw_materials', 'labor')
        if kwargs_items is None:
            kwargs_items = {key: {} for key in opex_items}

        cost_raw = np.asarray(cost_raw)

        # ---------- Heat duties
        # Energy cost (USD/GJ)
        heat_exchange_cost = [14.12, 8.49, 4.77,  # refrigeration
                              0.378,  # water
                              4.54, 4.77, 5.66]  # steam

        heat_exchange_cost = np.array(heat_exchange_cost)

        duties, map_duties = self.GetDuties(full_output=True,
                                            **kwargs_items.get('duties', {}))
        map_duties += 3

        duty_unit_cost = np.zeros_like(map_duties, dtype=np.float64)
        for ind, row in enumerate(map_duties):
            duty_unit_cost[ind] = heat_exchange_cost[row]

        duty_cost = np.abs(duties)*1e-9 * duty_unit_cost

        # ---------- Raw materials
        raw_materials = self.GetRawMaterials(
            include_holdups, steady_raw, **kwargs_items.get('raw_materials',
                                                            {}))
        raw_cost = cost_raw * raw_materials

        # ---------- Labor
        labor_cost = self.GetLabor(**kwargs_items.get('labor', {}))

        if lumped:
            pass
        else:
            return duty_cost, raw_cost, labor_cost

    def CreateStatsObject(self, alpha=0.95):
        statInst = StatisticsClass(self.ParamInst, alpha=alpha)
        return statInst
