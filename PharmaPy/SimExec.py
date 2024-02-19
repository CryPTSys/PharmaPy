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
from PharmaPy.Results import SimulationResult, flatten_dict_fields, get_name_object

from PharmaPy.Commons import trapezoidal_rule, check_steady_state
from PharmaPy.CheckModule import check_modeling_objects

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

    def SolveFlowsheet(self, kwargs_run=None, pick_units=None, verbose=True,
                       steady_state_di=None, tolerances_ss=None, ss_time=0):

        if kwargs_run is None:
            kwargs_run = {}

        if steady_state_di is None:
            steady_state_di = {}

        if pick_units is None:
            pick_units = self.execution_names

        if tolerances_ss is None:
            tolerances_ss = {}

        time_processing = {}

        # Run loop
        connections = {}
        count = 1

        # ss_time = 0
        for ind, name in enumerate(self.execution_names):
            instance = getattr(self, name)

            if name in pick_units:
                self.uos_instances[name] = instance
                check_modeling_objects(instance, name)

                if verbose:
                    print()
                    print('{}'.format('-'*30))
                    print('Running {}'.format(name))
                    print('{}'.format('-'*30))
                    print()

                kwargs_uo = kwargs_run.get(name, {})

                if name in steady_state_di:
                    kw_ss = steady_state_di[name]

                    tau = 0
                    if hasattr(instance, '_get_tau'):
                        tau = instance._get_tau()

                    ss_time += tau

                    if instance.__class__.__name__ == 'Mixer':
                        pass
                    else:
                        defaults = {'time_stop': ss_time,
                                    # 'threshold': 1e-6,
                                    'tau': tau}

                        for key, val in defaults.items():
                            kw_ss.setdefault(key, val)

                        ss_event = {'callable': check_steady_state,
                                    'num_conditions': 1,
                                    'event_name': 'steady_state',
                                    'kwargs': kw_ss
                                    }

                        # instance.state_event_list = [ss_event]
                        instance.state_event_list.append(ss_event)
                        kwargs_uo['any_event'] = False

                # check_modeling_objects(instance, name)
                instance.solve_unit(**kwargs_uo)

                uo_type = instance.__module__
                if uo_type != 'PharmaPy.Containers':
                    instance.flatten_states()

                if verbose:
                    print()
                    print('Done!')
                    print()

                # Create connection object if needed
                neighbors = self.graph[name]
                if len(neighbors) > 0 and self.execution_names[ind + 1] in pick_units:
                    uo_next = self.execution_names[ind + 1]
                    connection = Connection(
                        source_uo=getattr(self, name),
                        destination_uo=getattr(self, uo_next))

                    conn_name = 'CONN%i' % count
                    connections[conn_name] = connection

                    connection.transfer_data()

                    count += 1

                # Processing times
                if hasattr(instance.result, 'time'):
                    time_prof = instance.result.time
                    time_processing[name] = time_prof[-1] - time_prof[0]

            # instance is already solved, pass data to connection
            elif isinstance(instance.outputs, dict):
                connection = Connection(
                    source_uo=getattr(self, name),
                    destination_uo=getattr(self,
                                           self.execution_names[ind + 1]))

                conn_name = 'CONN%i' % count
                connections[conn_name] = connection

                connection.transfer_data()

                count += 1

        self.time_processing = time_processing

        self.result = SimulationResult(self)
        self.connections = connections

    def SetParamEstimation(self, x_data, y_data=None, y_spectra=None,
                           fit_spectra=False,
                           wrapper_kwargs=None,
                           phase_modifiers=None, control_modifiers=None,
                           pick_unit=None, **inputs_paramest):
        """
        Set parameter estimation using the aggregated unit operation to a
        simulation object

        Parameters
        ----------
        x_data : TYPE
            DESCRIPTION.
        y_data : TYPE, optional
            DESCRIPTION. The default is None.
        spectra : TYPE, optional
            DESCRIPTION. The default is None.
        fit_spectra : TYPE, optional
            DESCRIPTION. The default is False.
        wrapper_kwargs : TYPE, optional
            DESCRIPTION. The default is None.
        phase_modifiers : dict, optional
            Dictionary containing values to be set to the initial state
            of a phase for each experiment. Keys of 'phase_modifiers'
            must be experiment names, and fields must be dictionaries
            with keys matching the fields used to create a PharmaPy phase.
            An example for a reactor would be:

                my_modifier = {
                    'exp_1': {'temp': 300, 'mole_frac': [...]},
                    'exp_2': {'temp': 320, 'mole_frac': [...]}}

            For multi-phase systems such as crystallizer, an additional layer
            is needed to indicate which phase is being modified, e.g.

                my_modifier = {
                    'exp_1': {'Liquid_1': {'temp': 300, 'mole_frac': [...]}},
                    'exp_2': {'Liquid_1': {'temp': 320, 'mole_frac': [...]}, 'Solid_1': {'distrib':}}
                    }

            The default is None.
        control_modifiers : dict, optional
            Dictionary containing arguments to be passed to a control function.
            For instance, for a crystallizer with a temperature control with
            signature my_control(time, temp_init, ramp):

                my_modifier = {'temp': {'args': (320, -0.2)}}

            The default is None.
        pick_unit : TYPE, optional
            DESCRIPTION. The default is None.
        **inputs_paramest : TYPE
            DESCRIPTION.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # self.LoadUOs()

        if len(self.graph) == 1:
            target_unit = getattr(self, list(self.graph.keys())[0])
            # target_unit.reset_states = True
        else:
            if pick_unit is None:
                raise RuntimeError("Two or more unit operations detected. "
                                   "Select one using the 'pick_unit' argument")
            else:
                pass  # remember setting reset_states to True!!

        if phase_modifiers is None:
            if isinstance(x_data, dict):
                phase_modifiers = {key: {} for key in x_data}
            else:
                phase_modifiers = {}

        if control_modifiers is None:
            if isinstance(x_data, dict):
                control_modifiers = {key: {} for key in x_data}
            else:
                control_modifiers = {}

        if wrapper_kwargs is None:
            wrapper_kwargs = {}

        if isinstance(x_data, dict):
            kwargs_wrapper = {
                key: {'modify_phase': phase_modifiers[key],
                      'modify_controls': control_modifiers[key]}
                for key in x_data}

            for di in kwargs_wrapper.values():
                di.update({'run_args': wrapper_kwargs})
        else:
            kwargs_wrapper = {'modify_phase': phase_modifiers,
                              'modify_controls': control_modifiers}

            kwargs_wrapper['run_args'] = wrapper_kwargs

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
                param_seed=param_seed, time_data=x_data, y_spectra=y_spectra,
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

    def get_from_phases(self, phases, fields):
        if phases.__module__ == 'PharmaPy.MixedPhases':
            phases = phases.Phases
        else:
            phases = [phases]

        out = {}
        for phase in phases:
            di = {}
            for field in fields:
                di[field] = getattr(phase, field)

            name_phase = get_name_object(phase)
            out[name_phase] = di

        return out

    def get_raw_inlets(self, uo, basis='mass'):
        if hasattr(uo, 'Inlet'):
            if isinstance(uo.Inlet, dict):
                inlets = uo.Inlet
            else:
                inlets = [uo.Inlet]
        elif uo.__class__.__name__ == 'Mixer':
            inlets = uo.Inlets
        else:
            inlets = [None]

        if not isinstance(inlets, dict):
            inlets = {'Inlet_%i' % num: obj for num, obj in enumerate(inlets)}

        raws = {key: val for key, val in inlets.items()
                if val is not None and val.y_upstream is None}

        # inlets = [inlet for inlet in inlets
        #           if inlet is not None and inlet.y_upstream is None]

        out = {}

        for name, inlet in raws.items():
            if inlet.__class__.__name__ == 'PharmaPy.MixedPhases':
                streams = inlet.Phases
            else:
                streams = [inlet]

            di = {}
            for stream in streams:
                fields = ['temp', 'pres']

                name_stream = get_name_object(stream)

                di[name_stream] = {}

                dens = stream.getDensity(basis=basis)

                if uo.oper_mode == 'Batch':
                    if basis == 'mass':
                        total = stream.mass
                    elif basis == 'mole':
                        total = stream.moles
                elif inlet.DynamicInlet is None:
                    time = uo.result.time[-1] - uo.result.time[0]
                    if basis == 'mass':
                        flow = inlet.mass_flow
                        total = flow*time

                        di[name_stream] = {'mass': total}
                        fields += ['mass_frac', 'mass_flow', 'vol_flow']

                    else:
                        flow = inlet.mole_flow
                        total = flow*time

                        di[name_stream] = {'moles': total}
                        fields += ['mole_frac', 'mole_flow', 'vol_flow']

                else:
                    time = uo.result.time
                    inputs = uo.Inlet.DynamicInlet.evaluate_inputs(time)

                    if basis == 'mass':
                        if 'mass_flow' in inputs:
                            flow = inputs['mass_flow']
                        else:
                            flow = inputs['mole_flow'] * inlet.mw_av / 1000

                        total = trapezoidal_rule(time, flow)

                        di[name_stream] = {'mass': total}

                        fields += ['mass_frac']

                    elif basis == 'mole':
                        if 'mole_flow' in inputs:
                            flow = inputs['mole_flow']
                        else:
                            flow = inputs['mass_flow'] / inlet.mw_av * 1000

                        total = trapezoidal_rule(time, flow)

                        di[name_stream] = {'moles': total}
                        fields += ['mole_frac']

                vol = total / dens
                if basis == 'mole':
                    vol *= 1/1000

                di[name_stream]['vol'] = vol

            from_inlet = self.get_from_phases(inlet, fields)

            for key in from_inlet:
                di[key].update(from_inlet[key])

            out[name] = di

        return out

    def get_holdup(self, uo, basis='mass'):
        out = {}

        if hasattr(uo, '__original_phase__'):
            phases = uo.__original_phase__

            if basis == 'mass':
                fields = ['mass', 'mass_frac']
            elif basis == 'mole':
                fields = ['moles', 'mole_frac']

            fields += ['temp', 'pres', 'vol']

            if not phases.transferred_from_uo:
                out = self.get_from_phases(phases, fields)
                out = {'Initial_holdup': out}

        return out

    def GetRawMaterials(self, basis='mass', totals=True, steady_state=False):

        out = {}
        for name, uo in self.uos_instances.items():
            out[name] = {}

            raw_inlets = self.get_raw_inlets(uo, basis=basis)
            raw_holdup = self.get_holdup(uo, basis=basis)

            for second in raw_inlets:  # flatten multidimensional states
                for third in raw_inlets[second]:
                    di_raw = flatten_dict_fields(raw_inlets[second][third],
                                                 index=self.NamesSpecies)
                    raw_inlets[second][third] = di_raw

            for second in raw_holdup:
                for third in raw_holdup[second]:
                    di_hold = flatten_dict_fields(raw_holdup[second][third],
                                                  index=self.NamesSpecies)
                    raw_holdup[second][third] = di_hold

            out[name].update(raw_inlets)
            out[name].update(raw_holdup)

        di_multiindex = {(i, j, k): out[i][j][k]
                         for i in out
                         for j in out[i]
                         for k in out[i][j]}

        if len(di_multiindex) == 0:
            raw_df = pd.DataFrame()
        else:
            multi_index = pd.MultiIndex.from_tuples(di_multiindex)
            raw_df = pd.DataFrame(list(di_multiindex.values()),
                                  index=multi_index)

            if totals:
                if basis == 'mass':
                    mass_frac = raw_df.filter(regex='mass_frac').values

                    mass = raw_df['mass'].values[:, np.newaxis]
                    mass_comp = mass_frac * mass

                    cols = ['mass_%s' % comp for comp in self.NamesSpecies]
                    cols = ['mass'] + cols

                    raw_df = pd.DataFrame(np.column_stack((mass, mass_comp)),
                                          columns=cols, index=raw_df.index)

                elif basis == 'moles':
                    mole_frac = raw_df.filter(regex='mole_frac').values
                    moles = raw_df['moles'].values[:, np.newaxis]
                    moles_comp = mole_frac * moles

                    cols = ['moles_%s' % comp for comp in self.NamesSpecies]
                    cols = ['moles'] + cols

                    raw_df = pd.DataFrame(np.column_stack((moles, moles_comp)),
                                          columns=cols, index=raw_df.index)

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
