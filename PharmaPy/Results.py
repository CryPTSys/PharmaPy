# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:38:23 2022

@author: dcasasor
"""


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
