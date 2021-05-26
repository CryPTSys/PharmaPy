#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:18:25 2021

@author: dcasasor
"""


class MetaModelingClass:
    def __init__(self, file_name, class_name, model_type='ODE',
                 name_states=None):

        self.file_name = file_name
        self.class_name = class_name

        self.model_type = model_type
        self.name_states = name_states

        self.method_arguments = {'unit_model': ('time', 'states'),
                                 'material_balances': ['time'],
                                 'energy_balances': ['time'],
                                 'solve_model': ('runtime=None', ),
                                 'retrieve_results': ('time', 'states')}

        if name_states is not None:
            self.method_arguments['material_balances'] += name_states
            self.method_arguments['energy_balances'] += name_states

    def __write_method(self, open_object, method_name, arg_names,
                       internals=None):

        arguments = ', '.join(arg_names)
        open_object.write(' ' * 4 + 'def {}(self, {}):\n\n'.format(
            method_name, arguments))

        if method_name == 'unit_model' and self.name_states is not None:
            if self.model_type == 'ODE' or self.model_type  == 'DAE':
                prefix = ''
            elif self.model_type == 'PDE':
                prefix = ':, '

            assign_snippet = [
                ' '*8 + 'dict_states = {}\n',
                ' '*8 + 'count = 0\n',
                ' '*8 + 'for name, idx in zip(self.acum_len, self.name_states):\n',
                ' '*12 + 'dict_states[name] = states[{}count:idx]\n'.format(prefix),
                ' '*12 + 'count = idx\n\n']

            open_object.write(' ' * 8 + '# Create a dictionary of states\n')
            open_object.writelines(assign_snippet)

            open_object.write(
                ' '*8 + 'material = self.material_balances(time, **dict_states)\n')
            open_object.write(
                ' '*8 + 'energy = self.energy_balances(time, **dict_states)\n\n')

            if self.model_type == 'ODE':
                concat_snippet = ' '*8 + 'balances = np.concatenate(material, energy)\n'
            else:
                concat_snippet = ' '*8 + 'balances = np.hstack(material, energy)\n'

            open_object.writelines(concat_snippet)

            open_object.write(' ' * 8 + 'return balances\n\n')

        elif method_name == 'solve_model' and self.name_states is not None:
            open_object.write(' '*8 + '# Change these nums accordingly\n')

            for state in self.name_states:
                open_object.write(' ' * 8 + 'num_{} = 1\n'.format(
                    state))

            len_states = ['num_{}'.format(name) for name in self.name_states]
            len_states = ', '.join(len_states)

            open_object.write(
                '\n' + ' ' * 8 + 'len_states = np.array([{}])\n\n'.format(
                    len_states))

            open_object.write(
                ' '*8 + 'self.acum_len = len_states.cumsum()\n')

            open_object.write(
                ' '*8 + 'self.len_states = len_states\n\n')

            assimulo_snippet = [
                ' '*8 + 'problem = Explicit_Problem(self.unit_model)\n',
                ' '*8 + 'solver = CVode(problem)\n',
                '\n' + ' '*8 + 'time, states = solver.solve()\n\n']

            open_object.writelines(assimulo_snippet)

            open_object.write(' ' * 8 + 'return time, states\n\n')

        elif method_name == 'retrieve_results' and self.name_states is not None:
            if self.model_type == 'ODE':
                assign_outputs = [
                    ' '*8 + 'model_outputs = {}\n',
                    ' '*8 + 'count = 0\n',
                    ' '*8 + 'for idx, name in zip(self.acum_len, self.name_states):\n',
                    ' '*12 + 'model_outputs[name] = states[:, count:idx]\n',
                    ' '*12 + 'count = idx\n'
                    ]
            elif self.model_type == 'PDE':
                assign_outputs = [
                    ' '*8 + 'model_outputs = reorder_pde_outputs(states, self.num_nodes, self.len_states)']

            open_object.writelines(assign_outputs)

        else:
            open_object.write(' '*8 + 'return\n\n')

    def CreatePharmaPyTemplate(self):
        op = open(self.file_name, 'w')

        if self.model_type == 'DAE':
            solver = 'from assimulo.solvers import IDA\n\n'
        else:
            solver = 'from assimulo.solvers import CVode\n\n'

        if self.model_type == 'PDE':
            pharmapy_modules = [
                'from PharmaPy.Commons import reorder_pde_outputs\n\n\n']
        else:
            pharmapy_modules = []

        packages = [
            'import numpy as np\n',
            'from assimulo.problem import Explicit_Problem\n',
            solver]

        packages += pharmapy_modules

        op.writelines(packages)

        if self.model_type == 'PDE':
            args_init = ['num_nodes']
        elif self.model_type == 'ODE':
            args_init = []

        displayed_args_init = [' '*8 + 'self.{} = {}\n'.format(arg, arg)
                               for arg in args_init]

        args_init = ', '.join(args_init)

        op.write('class {}:\n'.format(self.class_name))
        op.write(' ' * 4 + 'def __init__(self, {}):\n\n'.format(args_init))
        op.writelines(displayed_args_init)

        op.write(' ' * 8 + 'return\n\n')

        # Nomenclature snippet
        name_states = list(self.name_states)
        nomenclature = [
            ' '*4 + 'def nomenclature(self):\n',
            ' '*8 + 'self.name_states = {}\n\n'.format(name_states)]

        op.writelines(nomenclature)

        for method, args in self.method_arguments.items():
            self.__write_method(op, method, args)

        op.close()


if __name__ == '__main__':
    name_file = 'test.py'
    name_class = 'GenericUO'
    states = ['temp', 'x_liq', 'y_gas', 'pres']

    meta_object = MetaModelingClass(name_file, name_class, name_states=states,
                                    model_type='ODE')
    meta_object.CreatePharmaPyTemplate()
