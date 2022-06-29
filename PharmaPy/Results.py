# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:38:23 2022

@author: dcasasor
"""

import numpy as np
import pandas as pd


def get_stream_info(obj, fields):
    if not isinstance(obj, (tuple, list)):
        obj = [obj]

    out = {}
    for ind, phase in enumerate(obj):
        stream_info = {}
        for field in fields:
            stream_info[field] = getattr(phase, field, None)

        phase_type = phase.__class__.__name__
        if phase_type in out:
            phase_type = phase_type + '_' + ind

        out[phase_type] = stream_info

    return out


def flatten_dict_fields(di, index=None):
    target_keys = []

    new_fields = {}
    for key in di:
        if isinstance(di[key], (tuple, list, np.ndarray)):
            val = di[key]
            target_keys.append(key)

            if index is None:
                suff = range(len(val))
            elif isinstance(index, (list, tuple)):
                suff = index
            else:
                suff = index[key]

            for ind, num in enumerate(val):
                new_fields['%s_%s' % (key, suff[ind])] = num

    for key in target_keys:
        di.pop(key)

    out = di | new_fields
    return out


def get_di_multiindex(di):

    out = {(i, j, k): di[i][j][k]
           for i in di.keys()
           for j in di[i].keys()
           for k in di[i][j].keys()}

    return out


def pprint(di, name_items, fields, str_out=True):
    """
    Create a table showing items with their respective fiels as columns

    Parameters
    ----------
    di : dict
        dictionary structured as:

            {'item_1': {'field_1':..., 'field_2':...},
             'item_2': {'field_1':..., 'field_2':...},
             ...}

    described_name : str
        name of described variable.
    fields : list of str
        name of fields to be taken from nested dictionaries.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """

    out = []

    form_header = []
    form_vals = []
    lens_header = []

    header = [name_items] + list(fields.keys())

    items = list(di.keys())

    # ---------- Lenghts
    # Lenght of first column
    max_lens = {name_items: max([len(name) for name in items])}

    # Lenght of remaining columns
    for field in fields:
        field_vals = [di[name].get(field, '') for name in items]

        for ind, val in enumerate(field_vals):
            if isinstance(val, list):
                if all([type(a) == str for a in val]):
                    field_vals[ind] = ', '.join(val)
                else:
                    field_vals[ind] = '%i, ..., %i' % (val[0], val[-1])

        len_vals = [len(repr(val)) for val in field_vals]
        max_lens[field] = max(len_vals)

    # All fields
    all_fields = {name_items: 's'} | fields
    for name in all_fields:
        le = max(len(name), max_lens[name]) + 2

        form_header.append("{:<%i}" % le)
        form_vals.append("{:<%i%s}" % (le, all_fields[name]))

        lens_header.append(le)

    form_header = ' '.join(form_header)
    form_vals = ' '.join(form_vals)

    len_headers = sum(lens_header)

    lines = '-' * len_headers
    out.append(lines)
    out.append(form_header.format(*header))
    out.append(lines)

    for name in di:
        field_vals = [di[name].get(field, '') for field in fields]

        for ind, val in enumerate(field_vals):
            if isinstance(val, list):
                if all([type(a) == str for a in val]):
                    field_vals[ind] = ', '.join(val)
                else:
                    field_vals[ind] = '%i, ..., %i' % (val[0], val[-1])

        item = form_vals.format(*([name] + field_vals))

        out.append(item)

    out.append(lines)

    if str_out:
        out = '\n'.join(out)

    return out


class DynamicResult:
    def __init__(self, di_states, di_fstates=None, **results):
        self.__dict__.update(**results)
        self.di_states = di_states
        self.di_fstates = di_fstates

    def __repr__(self):
        headers = {'dim': '', 'units': 's', 'index': 's'}

        str_states = pprint(self.di_states, 'states', headers)

        state_example = list(self.di_states.keys())[0]

        header = 'PharmaPy result object'
        lines = '-' * (len(header) + 2)

        header = '\n'.join([lines, header, lines + '\n\n'])

        explain = 'Fields shown in the tables below can be accessed as ' \
            'result.<field>, e.g. result.%s \n\n' % state_example

        top_text = header + explain

        if self.di_fstates is not None and len(self.di_fstates) > 0:
            head = {'dim': '', 'units': 's', 'index': 's'}
            str_fstates = pprint(self.di_fstates, 'f(states)', head)

            out_str = str_states + '\n\n' + str_fstates

        else:
            out_str = str_states

        out_str = top_text + out_str
        out_str += '\n\nTime vector can be accessed as result.time\n'

        return out_str


class SimulationResult:
    def __init__(self, sim):
        self.sim = sim

        # Create UO summary
        di_uos = {}

        names_uos = sim.execution_names

        headers = {'Diff eqns': 'd', 'Alg eqns': 'd',
                   'Model type': 's', 'PharmaPy type': 's'}

        for name in names_uos:
            states_di = getattr(getattr(sim, name), 'states_di', None)

            if states_di is not None:
                num_diff = []
                num_alg = []
                for var, di in states_di.items():
                    num = di.get('index', 1)

                    if isinstance(num, (list, tuple)):
                        num = len(num)

                    if di['type'] == 'diff':
                        num_diff.append(num)
                    elif di['type'] == 'alg':
                        num_alg.append(num)

                if sum(num_diff) == 0:
                    model_type = 'ALG'
                elif sum(num_diff) > 0 and sum(num_alg) > 0:
                    model_type = 'DAE'
                else:
                    model_type = 'ODE'

                di_uos[name] = {}

                pharmapy_type = getattr(sim, name).__class__.__name__

                di_uos[name]['Diff eqns'] = sum(num_diff)
                di_uos[name]['Alg eqns'] = sum(num_alg)
                di_uos[name]['Model type'] = model_type
                di_uos[name]['PharmaPy type'] = pharmapy_type

        out_uos = pprint(di_uos, 'Unit operation', headers, str_out=False)

        self.out_uos = out_uos

    def get_raw_objects(self):
        out = {}

        for name, uo in self.sim.uos_instances.items():
            # Inlets (flows)
            if hasattr(uo, 'Inlet'):
                if uo.Inlet is not None:
                    if uo.Inlet.y_upstream is None:
                        inlet = getattr(uo, 'Inlet_orig', getattr(uo, 'Inlet'))

                        if uo.oper_mode == 'Batch':
                            elapsed_time = 1
                        elif uo.Inlet.DynamicInlet is not None:
                            elapsed_time = uo.result.time
                        else:
                            elapsed_time = uo.timeProf[-1] - uo.timeProf[0]

                    else:
                        inlet = None
                        elapsed_time = None

            elif uo.__class__.__name__ == 'Mixer':
                inlet = []
                elapsed_time = []

                for inl in uo.Inlets:
                    if inl.y_upstream is None:
                        inlet.append(inl)

                        if uo.oper_mode == 'Batch':
                            time = 1
                        elif inl.DynamicInlet is None:
                            time_prof = uo.result.time
                            time = time_prof[-1] - time_prof[0]

                    elapsed_time.append(time)

            else:
                inlet = None
                elapsed_time = None

            out[name] = {'raw_streams': inlet, 'time_streams': elapsed_time}

            # Initial holdups
            if hasattr(uo, '__original_phase__'):
                orig = uo.__original_phase__
            else:
                orig = None

            out[name]['raw_holdups'] = orig

        return out

    def GetStreamTable(self, basis='mass'):
        uo_dict = self.sim.uos_instances

        # TODO: include inlets and holdups in the stream table
        inlet_di = self.get_raw_objects()

        if basis == 'mass':
            fields_phase = ['temp', 'pres', 'mass', 'vol', 'mass_frac']
            fields_stream = ['temp', 'pres', 'mass_flow', 'vol_flow', 'mass_frac']

            frac_preffix = 'w_{}'
        elif basis == 'mole':
            fields_phase = ['temp', 'pres', 'moles', 'vol', 'mole_frac']
            fields_stream = ['temp', 'pres', 'mole_flow', 'vol_flow', 'mole_frac']

            frac_preffix = 'x_{}'

        info = {}
        for ind, name in enumerate(uo_dict):
            matter_obj = uo_dict[name].Outlet
            info[name] = {}

            if 'Stream' in matter_obj.__class__.__name__:
                fields = fields_stream
            else:
                fields = fields_phase

            if matter_obj.__module__ == 'PharmaPy.MixedPhases':
                matter_obj = matter_obj.Phases

            raw_inlets = inlet_di[name]['raw_streams']
            if raw_inlets is not None:
                entries = get_stream_info(raw_inlets, fields)
                entries = {key: flatten_dict_fields(val, self.sim.NamesSpecies)
                           for key, val in entries.items()}

                info[name]['Raw inlets'] = entries

            # raw_holdup = inlet_di[name]['raw_holdups']
            # if raw_holdup is not None:
            #     info[name]['Initial holdup'] = get_stream_info(raw_holdup, fields)

            entries = get_stream_info(matter_obj, fields)
            entries = {key: flatten_dict_fields(val, self.sim.NamesSpecies)
                       for key, val in entries.items()}
            info[name]['Outlet'] = entries

        di_multiindex = get_di_multiindex(info)
        mux = pd.MultiIndex.from_tuples(di_multiindex.keys())

        stream_table = pd.DataFrame(list(di_multiindex.values()), index=mux)

        return stream_table

    def __repr__(self):
        # Welcome message
        welcome = 'Welcome to PharmaPy'
        len_header = len(welcome) + 2
        lines = '-' * len_header

        names_uos = self.sim.execution_names
        out = [lines, welcome, lines + '\n']

        # Flowsheet ASCII diagram (if the graph is simple)
        if names_uos is not None:
            is_simple = all([a < 2 for a in list(self.sim.in_degree.values())])

            if is_simple:

                flow_diagram = ' --> '.join(names_uos)

                out += ['Flowsheet structure:', flow_diagram + '\n']

        # Include UOs table
        out_str = '\n'.join(out + self.out_uos)

        return out_str
