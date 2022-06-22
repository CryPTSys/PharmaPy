# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:38:23 2022

@author: dcasasor
"""


def pprint(di, headers):
    out = []

    field_names = list(headers.keys())

    form_header = []
    form_vals = []
    lens_header = []

    state_names = list(di.keys())
    max_lens = {field_names[0]: max([len(name) for name in state_names])}

    for name in field_names[1:]:
        items = [len(repr(di[st][name])) for st in state_names]
        max_lens[name] = max(items)

    for ind, (header, typ) in enumerate(headers.items()):
        le = max(len(header), max_lens[header]) + 2

        form_header.append("{:<%i}" % le)
        form_vals.append("{:<%i%s}" % (le, typ))

        lens_header.append(le)

    form_header = ' '.join(form_header)
    form_vals = ' '.join(form_vals)

    len_headers = sum(lens_header)

    lines = '-' * len_headers
    out.append(lines)
    out.append(form_header.format(*headers))
    out.append(lines)

    for name in di:
        field_vals = [di[name][field] for field in field_names[1:]]
        item = form_vals.format(*([name] + field_vals))

        out.append(item)

    out.append(lines)
    out = '\n'.join(out)

    return out


class DynamicResult:
    def __init__(self, di_states, di_fstates=None, **results):
        self.__dict__.update(**results)
        self.di_states = di_states
        self.di_fstates = di_fstates

    def __repr__(self):
        headers = {'states': 's', 'dim': '', 'units': 's'}

        str_states = pprint(self.di_states, headers)

        if self.di_fstates is not None and len(self.di_fstates) > 0:
            head = {'f(states)': 's', 'dim': '', 'units': 's'}
            str_fstates = pprint(self.di_fstates, head)

            out_str = str_states + '\n\n' + str_fstates

        else:
            out_str = str_states

        out_str += '\n\nTime vector can be accessed as result.time\n'

        return out_str
