# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:36:35 2020

@author: dcasasor
"""

from PharmaPy.NameAnalysis import NameAnalyzer, get_dict_states
from PharmaPy.Interpolation import local_newton_interpolation

from scipy.interpolate import CubicSpline

import numpy as np
import copy


def interpolate_inputs(time, t_inlet, y_inlet, **kwargs_interp_fn):
    if isinstance(time, (float, int)):
        # Assume steady state for extrapolation
        time = min(time, t_inlet[-1])

        y_interp = local_newton_interpolation(time, t_inlet, y_inlet,
                                              **kwargs_interp_fn)
    else:
        interpol = CubicSpline(t_inlet, y_inlet, **kwargs_interp_fn)
        flags_extrapol = time > t_inlet[-1]

        if any(flags_extrapol):
            time_interpol = time[~flags_extrapol]
            y_interp = interpol(time_interpol)

            if y_inlet.ndim == 1:
                y_extrap = np.tile(y_interp[-1], sum(flags_extrapol))
                y_interp = np.concatenate((y_interp, y_extrap))
            else:
                y_extrap = np.tile(y_interp[-1], (sum(flags_extrapol), 1))
                y_interp = np.vstack((y_interp, y_extrap))
        else:
            y_interp = interpol(time)

    return y_interp


def get_input_dict(input_data, name_dict):

    dict_out = {}
    count = 0
    for phase, states in name_dict.items():
        names = list(states.keys())
        lens = list(states.values())

        acum_len = np.cumsum(lens)
        if len(acum_len) > 1:
            acum_len = acum_len[:-1]

        if input_data.ndim == 1:
            splitted = np.split(input_data[count:], acum_len, axis=0)
            splitted = [ar[0] if len(ar) == 1 else ar for ar in splitted]
        else:
            splitted = np.split(input_data[:, count:], acum_len, axis=1)

            for ind, val in enumerate(splitted):
                if val.shape[1] == 1:
                    splitted[ind] = val.flatten()

        for ind, elem in enumerate(splitted):
            if isinstance(elem, np.ndarray) and len(elem) == 0:
                splitted[ind] = np.zeros(lens[ind])

        count += acum_len[-1]

        dict_out[phase] = dict(zip(names, splitted))

    return dict_out


def get_missing_field(dim_state, n_times):
    if n_times == 1:
        if dim_state == 1:
            out = 0
        else:
            out = np.zeros(dim_state)
    elif n_times > 1:
        if dim_state == 1:
            out = np.zeros(n_times)

        else:
            out = np.zeros((n_times, dim_state))

    return out


def get_remaining_states(dict_states_in, stream, inlets, time):
    di_out = {}
    time = np.atleast_1d(time)
    for phase, di in dict_states_in.items():
        di_out[phase] = {}
        if 'inlet' in phase.lower():
            for state in di:
                if state not in inlets[phase]:
                    field = getattr(stream, state, None)

                    if field is None:
                        field = get_missing_field(
                            di[state], len(time))

                    elif len(time) > 1:
                        field = np.outer(np.ones_like(time), field)
                        if field.shape[1] == 1:
                            field = field.flatten()

                    di_out[phase][state] = field
        else:
            for state in di:
                if state not in inlets[phase]:
                    sub_phase = getattr(stream, phase)
                    field = getattr(sub_phase, state, None)

                    if field is None:
                        field = get_missing_field(
                            di[state], len(time))

                    di_out[phase][state] = field
    return di_out


def get_inputs_new(time, stream, dict_states_in, **kwargs_interp):
    """
    Get inputs based on stream(s) object and names of inlet states

    Parameters
    ----------
    time : float
        evaluation time [s].
    stream : PharmaPy.Stream
        stream object.
    names_in : dict
        dictionary with names of inlet states to the destination unit operation
        as keys and dimension of the state as values.
    **kwargs_interp : keyword arguments
        arguments to be passed to the particular interpolation function.

    Returns
    -------
    inputs : dict
        dictionary with states.

    """

    if stream.DynamicInlet is not None:
        inputs = stream.DynamicInlet.evaluate_inputs(time, **kwargs_interp)
        inputs = {'Inlet': inputs}

    elif stream.y_upstream is not None and stream.time_upstream is not None:
        t_inlet = stream.time_upstream
        y_inlet = stream.y_inlet

        ins = {}
        for key, val in y_inlet.items():
            ins[key] = interpolate_inputs(time, t_inlet, val,
                                          **kwargs_interp)

        inputs = {}
        for phase, names in dict_states_in.items():
            inputs[phase] = {}
            for key, vals in ins.items():
                if key in names:
                    inputs[phase][key] = vals

    elif stream.y_upstream is not None:
        inputs = {}

        for phase, names in dict_states_in.items():
            inputs[phase] = {}
            for key, vals in stream.y_inlet.items():
                if key in names:
                    if isinstance(time, np.ndarray) and len(time) > 1:
                        vals = np.outer(np.ones_like(time), vals)

                        if vals.shape[1] == 1:
                            vals = vals.flatten()

                    inputs[phase][key] = vals

    else:
        inputs = {obj: {} for obj in dict_states_in.keys()}

    remaining = get_remaining_states(dict_states_in, stream, inputs, time)

    for key in dict_states_in:
        inputs[key] = {**inputs[key], **remaining[key]}

    return inputs


def get_inputs(time, uo, num_species, num_distr=0):
    Inlet = getattr(uo, 'Inlet', None)

    names_upstream = uo.names_upstream
    names_states_in = uo.names_states_in
    bipartite = uo.bipartite
    if Inlet is None:
        input_dict = {}

        return input_dict

    elif Inlet.y_upstream is None or len(Inlet.y_upstream) == 1:
        # this internally calls the DynamicInput object if not None
        input_dict = Inlet.evaluate_inputs(time)

        for name in names_states_in:
            if name not in input_dict.keys():
                val = getattr(Inlet, name, None)
                if val is None:  # search in subphases inside Inlet
                # if hasattr(uo, 'states_in_phaseid'):
                    obj_id = uo.states_in_phaseid[name]
                    instance = getattr(Inlet, obj_id)
                    val = getattr(instance, name)

                input_dict[name] = val

    else:
        all_inputs = Inlet.InterpolateInputs(time)
        input_upstream = get_dict_states(names_upstream, num_species,
                                         num_distr, all_inputs)

        input_dict = {}
        for key in names_states_in:
            val = input_upstream.get(bipartite[key])
            if val is None:
                val = uo.input_defaults[key]

            input_dict[key] = val

    return input_dict


def topological_bfs(graph):
    in_degree = {}
    for node, neighbors in graph.items():
        in_degree.setdefault(node, 0)
        for n in neighbors:
            in_degree[n] = in_degree.get(n, 0) + 1

    path = []

    no_incoming = {node for node, count in in_degree.items() if count == 0}

    while no_incoming:
        v = no_incoming.pop()
        path.append(v)
        for adj in graph.get(v, []):
            in_degree[adj] -= 1

            if in_degree[adj] == 0:
                no_incoming.add(adj)

    return in_degree, path


def convert_str_flowsheet(flowsheet):
    seq = [a.strip() for a in flowsheet.split('-->')]

    out = {}
    num_uos = len(seq)
    for ind in range(num_uos - 1):
        out[seq[ind]] = [seq[ind + 1]]

    out[seq[num_uos - 1]] = []
    return out


class Connection:
    def __init__(self, source_uo, destination_uo):

        self.source_uo = source_uo
        self.destination_uo = destination_uo

    def transfer_data(self):
        self.FeedConnection()
        self.ConvertUnits()
        self.PassPhases()

    def FeedConnection(self):
        self.Matter = self.source_uo.Outlet
        if isinstance(self.Matter, dict):
            self.Matter = self.Matter[self.source_uo.default_output]

        self.num_species = self.Matter.num_species

        self.Matter.y_upstream = self.source_uo.outputs

        time_prof = self.source_uo.result.time

        if self.source_uo.is_continuous:
            self.Matter.time_upstream = time_prof
        else:
            self.Matter.time_upstream = time_prof[-1]

    def ConvertUnits(self):
        mode_source = self.source_uo.oper_mode
        mode_dest = self.destination_uo.oper_mode

        flow_flag = (mode_source == 'Continuous' and mode_dest != 'Batch')
        btf_flag = self.source_uo.__class__.__name__ == 'BatchToFlowConnector'

        if flow_flag:
            states_up = self.source_uo.names_states_out

            class_destination = self.destination_uo.__class__.__name__
            if class_destination == 'DynamicCollector':
                if self.source_uo.__class__.__name__ == 'MSMPR':
                    states_down = self.destination_uo.names_states_in['crystallizer']
                else:
                    states_down = self.destination_uo.names_states_in['liquid_mixer']

            else:
                states_down = self.destination_uo.names_states_in
            
            # if hasattr(self.Matter, 'moments'):
            #     num_distr = len(self.Matter.moments)
            
            # elif hasattr(self.Matter, 'distrib'):
            #     num_distr = len(self.Matter.distrib)
            
            if 'mu_n' in states_up:
                num_distr = len(self.Matter.moments)
            elif 'distrib' in states_up:
                num_distr = len(self.Matter.distrib)
            else:
                num_distr = 0
                
            name_analyzer = NameAnalyzer(
                states_up, states_down, self.num_species,
                num_distr)

            # Convert units and pass states to self.Matter
            converted_states = name_analyzer.convertUnits(self.Matter)
            self.Matter.y_inlet = converted_states

        elif btf_flag:
            states_up = self.source_uo.names_states_out

            class_destination = self.destination_uo.__class__.__name__
            if class_destination == 'DynamicCollector':
                if self.source_uo.__class__.__name__ == 'MSMPR':
                    states_down = self.destination_uo.names_states_in['crystallizer']
                else:
                    states_down = self.destination_uo.names_states_in['liquid_mixer']

            else:
                states_down = self.destination_uo.names_states_in

            name_analyzer = NameAnalyzer(
                states_up, states_down, self.num_species,
                len(getattr(self.Matter, 'distrib', []))
                )

            # Convert units and pass states to self.Matter
            converted_states = name_analyzer.convertUnits(self.Matter)
            self.Matter.y_inlet = converted_states

    def PassPhases(self):

        class_destination = self.destination_uo.__class__.__name__
        mode_dest = self.destination_uo.oper_mode
        transfered_matter = copy.deepcopy(self.Matter)

        transfered_matter.transferred_from_uo = True

        if class_destination == 'Mixer':
            self.destination_uo.Inlets = transfered_matter

        elif mode_dest == 'Batch':
            self.destination_uo.Phases = transfered_matter
            # if class_destination == 'BatchToFlowConnector':
            #     self.destination_uo.names_states_out = self.source_uo.names_states_out

        elif mode_dest == 'Semibatch':
            if class_destination == 'DynamicCollector':
                self.destination_uo.Inlet = transfered_matter
                self.destination_uo.material_from_upstream = True

                if self.source_uo.__module__ == 'PharmaPy.Crystallizers':
                    self.destination_uo.KinCryst = self.source_uo.Kinetics
                    self.destination_uo.kwargs_cryst = {
                        'target_ind': self.source_uo.target_ind,
                        'target_comp': self.source_uo.target_comp,
                        'scale': self.source_uo.scale}

            elif self.destination_uo.Phases is None:
                self.destination_uo.Phases = transfered_matter
                self.destination_uo.material_from_upstream = True

        elif mode_dest == 'Continuous':  # Continuous
            # Transfering from batch to continuous (how to approach this?)
            if self.source_uo.oper_mode != 'Continuous':
                pass
                # TODO: big TODO. We need to define how Batch/Semibatch
                # followed by continuous will be handled. The most practical
                # approach would be to solve thhe the downstream continuous
                # section for a period of time such as the material from the
                # last discontinuous UO is depleted, as stated in the paper.
                # Reference date: (2022/06/28)

            if class_destination == 'DynamicExtractor':
                self.destination_uo.Inlet = {'feed': transfered_matter}
            else:
                self.destination_uo.Inlet = transfered_matter
