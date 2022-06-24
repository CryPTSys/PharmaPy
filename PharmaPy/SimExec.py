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

from PharmaPy.Connections import Connection, convert_str_flowsheet, topological_bfs
from PharmaPy.Errors import PharmaPyNonImplementedError
from PharmaPy.Results import SimulationResult

from PharmaPy.Commons import trapezoidal_rule, check_steady_state

import time


class SimulationExec:
    def __init__(self, pure_path, flowsheet):

        # Interfaces
        thermo_instance = ThermoPhysicalManager(pure_path)
        self.NamesSpecies = thermo_instance.name_species

        # Outputs
        self.StreamTable = None

        self.uos_instances = {}  # TODO: check this under the new graph implem
        self.oper_mode = []

        if isinstance(flowsheet, dict):
            graph = flowsheet
        elif isinstance(flowsheet, str):
            graph = convert_str_flowsheet(flowsheet)

        self.graph = graph
        self.in_degree, self.execution_names = topological_bfs(graph)

        if len(self.execution_names) < len(self.graph):
            raise PharmaPyNonImplementedError(
                "Provided flowsheet contains recycle stream(s)")

    def connect_flowsheet(self, graph):
        count = 1

        connections = []
        for node, neighbors in graph.items():
            source = getattr(self, node)

            for adj in neighbors:
                dest = getattr(self, adj)
                connection = Connection(source_uo=source, destination_uo=dest)

                conn_name = 'CONN%i' % count
                setattr(self, conn_name, connection)

                connections.append(connection)

                count += 1

        return connections

    def LoadUOs(self):
        uos_modules = ('Reactors', 'Crystallizers', 'Containers',
                       'Evaporators', 'SolidLiquidSep', 'Drying_Model')

        modules_ids = ['PharmaPy.' + elem for elem in uos_modules]

        # if self.execution_names is None:

        for name in self.execution_names:
        # for key, value in self.__dict__.items():
        #     type_val = getattr(value, '__module__', None)

        #     if type_val in modules_ids:
            value = getattr(self, name)
            value.id_uo = name
            value.name_species = self.NamesSpecies

                # self.uos_instances[key] = value
                # self.oper_mode.append(value.oper_mode)

    def SolveFlowsheet(self, kwargs_run=None, pick_units=None, verbose=True,
                       uos_steady_state=None, tolerances_ss=None, ss_time=0,
                       kwargs_ss=None):

        if len(self.uos_instances) == 0:
            self.LoadUOs()
            # self.LoadConnections()
            connections = self.connect_flowsheet(self.graph)

        # Pick specific units, if given
        if kwargs_run is None:
            if pick_units is None:
                keys = self.uos_instances.keys()
            else:
                keys = pick_units

            kwargs_run = {key: {} for key in keys}

        if kwargs_ss is None:
            kwargs_ss = {}

        execution_names = self.execution_names
        execution_uos = [getattr(self, name) for name in execution_names]

        if pick_units is not None:
            ordered_names = []
            ordered_uos = []

            for name in execution_names:
                if name in pick_units:
                    ordered_names.append(name)

                    idx = execution_names.index(name)
                    ordered_uos.append(execution_uos[idx])

            execution_names = ordered_names
            execution_uos = ordered_uos


        if tolerances_ss is None:
            tolerances_ss = {}

        # Run loop
        for name, instance in zip(execution_names, execution_uos):
            if verbose:
                print()
                print('{}'.format('-'*30))
                print('Running {}'.format(name))
                print('{}'.format('-'*30))
                print()

            for conn in connections:  # TODO: this is confusing
                if conn.destination_uo is instance:
                    conn.ReceiveData()  # receive phases from upstream uo
                    conn.TransferData()

            kwargs_uo = kwargs_run.get(name, {})

            tau = 0
            if hasattr(instance, '_get_tau'):
                tau = instance._get_tau()

            ss_time += tau

            if uos_steady_state is not None:
                if name in uos_steady_state:
                    if instance.__class__.__name__ == 'Mixer':
                        pass
                    else:
                        tolerances = tolerances_ss.get(name, 1e-6)

                        kw_ss = kwargs_ss.get(name, None)

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

        time_processing = np.zeros(len(execution_names))
        for ind, uo in enumerate(execution_names):
            if hasattr(uo, 'dynamic_result'):
                time_prof = uo.dynamic_result.time
                time_processing[ind] = time_prof[-1] - time_prof[0]
            elif hasattr(uo, 'timeProf'):
                time_processing[ind] = uo.timeProf[-1] - uo.timeProf[0]

        self.time_processing = time_processing

        self.result = SimulationResult(self)


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

    # def SetParamEstimation(self, x_data, y_data=None, param_seed=None,
    #                        wrapper_kwargs=None,
    #                        spectra=None,
    #                        fit_spectra=False, global_analysis=True,
    #                        phase_modifiers=None, control_modifiers=None,
    #                        measured_ind=None, optimize_flags=None,
    #                        jac_fun=None,
    #                        covar_data=None,
    #                        pick_unit=None):

    def SetParamEstimation(self, x_data, y_data=None, spectra=None,
                           fit_spectra=False,
                           wrapper_kwargs=None,
                           phase_modifiers=None, control_modifiers=None,
                           pick_unit=None, **inputs_paramest):

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

        # Get 1D array of parameters from the UO class
        param_seed = inputs_paramest.pop('param_seed', None)
        if param_seed is not None:
            target_unit.Kinetics.set_params(param_seed)

        if hasattr(target_unit, 'Kinetics'):
            param_seed = target_unit.Kinetics.concat_params()
        else:
            param_seed = target_unit.params

        name_params = inputs_paramest.get('name_params')

        if name_params is None:
            name_params = []
            for ind, logic in enumerate(target_unit.mask_params):
                if logic:
                    if hasattr(target_unit, 'Kinetics'):
                        name_params.append(
                            target_unit.Kinetics.name_params[ind])
                    else:
                        name_params.append(target_unit.name_params[ind])

        name_states = target_unit.states_uo

        inputs_paramest['name_states'] = name_states
        inputs_paramest['name_params'] = name_params

        # Instantiate parameter estimation
        if fit_spectra:
            self.ParamInst = MultipleCurveResolution(
                target_unit.paramest_wrapper,
                param_seed=param_seed, x_data=x_data, spectra=spectra,
                kwargs_fun=kwargs_wrapper,
                **inputs_paramest)
        else:
            self.ParamInst = ParameterEstimation(
                target_unit.paramest_wrapper,
                param_seed=param_seed, x_data=x_data, y_data=y_data,
                kwargs_fun=kwargs_wrapper,
                **inputs_paramest)

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
                            # time_inlets.append(uo.timeProf)
                            time_inlets.append(uo.dynamic_result.time)
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
                            if hasattr(uo, 'dynamic_result'):
                                time_prof = uo.dynamic_result.time
                            else:
                                time_prof = uo.timeProf

                            elapsed = time_prof[-1] - time_prof[0]
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

    # def __repr__(self):  # This is just a very primitive idea
    #     welcome = 'Welcome to PharmaPy'
    #     len_header = len(welcome) + 2
    #     lines = '-' * len_header
    #     out = [lines, welcome, lines]
    #     if self.execution_names is not None:
    #         is_simple = all([a < 2 for a in list(self.in_degree.values())])

    #         if is_simple:

    #             flow_diagram = ' --> '.join(self.execution_names)

    #             out += ['Flowsheet structure:', flow_diagram]

        # uo_header = 0
        # for name in self.execution_names:
        #     states_di = getattr(getattr(self, name), 'states_di')
        #     var_types = [di['type'] for di in states_di.values()]

        #     row = '\n' + name + var_types.count('diff') ''  + var_types


        #     return '\n'.join(out)
