# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:38:23 2022

@author: dcasasor
"""


class DynamicResult:
    def __init__(self, uo, **results):
        self.__dict__.update(**results)
        self.uo = uo

    def __repr__(self):
        di_states = self.uo.states_di

        out = []

        headers = {'states': 's', 'dim': '', 'units': 's'}
        field_names = list(headers.keys())[1:]

        form_header = []
        form_vals = []
        lens_header = []

        state_names = list(di_states.keys())
        max_lens = {'states': max([len(name) for name in state_names])}

        for name in field_names:
            items = [len(repr(di_states[st][name])) for st in state_names]
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

        for name in di_states:
            field_vals = [di_states[name][field] for field in field_names]
            item = form_vals.format(*([name] + field_vals))

            out.append(item)

        out.append(lines)
        out.append('\nTime vector can be accessed as result.time\n',)

        out = '\n'.join(out)

        return out
