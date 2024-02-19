#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:18:25 2021

@author: dcasasor
"""


class MetaModelingClass:
    def __init__(self, file_name, class_name, model_type='ODE',
                 oper_mode='batch', name_states=None, has_stages=False):

        self.file_name = file_name
        self.class_name = class_name

        self.model_type = model_type
        self.oper_mode = oper_mode
        self.name_states = name_states
        self.has_stages = has_stages

        self.method_arguments = {
            'Phases': ('phases', ),
            'unit_model': ('time', 'states'),
            'material_balances': ['time'],
            'energy_balances': ['time'],
            'solve_model': ('runtime', ),
            'retrieve_results': ('time', 'states')}

        if name_states is not None:
            self.method_arguments['material_balances'] += name_states
            self.method_arguments['energy_balances'] += name_states

    def __write_method(self, open_object, method_name, arg_names,
                       internals=None):

        arguments = ', '.join(arg_names)
        if method_name == 'Phases':
            phases_prop = [' ' * 4 + '@property\n',
                           ' ' * 4 + 'def Phases(self):\n',
                           ' ' * 8 + 'return self._Phases\n\n']

            phases_set = [' ' * 4 + '@Phases.setter\n',
                          ' ' * 4 + 'def {}(self, {}):\n\n'.format(
                              method_name, arguments),
                          ' ' * 8 + 'classify_phases(self)\n\n']

            open_object.writelines(phases_prop)
            open_object.writelines(phases_set)
        else:
            open_object.write(' ' * 4 + 'def {}(self, {}):\n\n'.format(
                method_name, arguments))

        if method_name == 'unit_model' and self.name_states is not None:
            if self.model_type == 'ODE' or self.model_type == 'DAE':
                prefix = ''
                assign_snippet = [
                ' '*8 + 'states_split = np.split(states, self.acum_len)\n',
                ' '*8 + 'dict_states = dict(zip(self.name_states, states_split))\n',
                '\n'
                ]
            elif self.model_type == 'PDE':
                prefix = ':, '
                assign_snippet = [
                ' '*8 + 'states_reord = states.reshape(-1, self.num_states)\n',
                ' '*8 + 'states_split = np.split(states_reord, self.acum_len, axis=1)\n\n',
                ' '*8 + 'dict_states = dict(zip(self.name_states, states_split))\n',
                '\n'
                ]

            if self.has_stages:
                reshape_lines = [
                    ' '*8 + 'states_reord = []\n',
                    ' '*8 + 'for array, num in zip(states_split, self.len_states_orig):\n',
                    ' '*12 + 'if num == 1:\n',
                    ' '*16 + 'states_reord.append(array)\n',
                    ' '*12 + 'else:\n',
                    ' '*16 + 'states_reord.append(array.reshape(-1, num))\n\n',
                    ' '*8 + 'states_split = states_reord\n']

                count = 1
                for line in reshape_lines:
                    assign_snippet.insert(count, line)
                    count += 1

            open_object.write(' ' * 8 + "'''This method will work by itself and does not need any user manipulation.\n")
            open_object.write(' ' * 8 + "Fill material and energy balances with your model.'''\n\n")
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
            open_object.write(' '*8 + '# If the model has stages, these nums '
                              'are the number of variables per plate, thus '
                              'a scalar variable would have num = 1\n')

            for state in self.name_states:
                open_object.write(' ' * 8 + 'num_{} = 1\n'.format(
                    state))

            len_states = ['num_{}'.format(name) for name in self.name_states]
            len_states = ', '.join(len_states)

            open_object.write(
                '\n' + ' ' * 8 + 'len_states = np.array([{}])\n'.format(
                    len_states))

            if self.has_stages:
                open_object.write(
                    '\n' + ' ' * 8 + 'self.len_states_orig = len_states.copy()\n')
                open_object.write(
                    '\n' + ' ' * 8 + 'len_states *= self.num_stages\n\n')

            open_object.write(
                ' '*8 + 'self.acum_len = len_states.cumsum()[:-1]\n')

            open_object.write(
                ' '*8 + 'self.len_states = len_states\n\n')

            open_object.write(
                ' '*8 + 'self.num_states = len_states.sum()\n\n')

            if self.model_type == 'ODE':
                init_states = [' ' * 8 + 'init_states = np.hstack(())\n\n']
            else:
                init_states = [' ' * 8 + 'init_states = np.hstack(())\n',
                               ' ' * 8 + 'init_states = init_states.T.ravel()\n\n']

            open_object.writelines(init_states)

            assimulo_snippet = [
                ' '*8 + 'problem = %s(self.unit_model)\n' % self.problem,
                ' '*8 + 'solver = %s(problem)\n' % self.solver,
                '\n' + ' '*8 + 'time, states = solver.solve(runtime)\n\n']

            open_object.writelines(assimulo_snippet)

            open_object.write(' ' * 8 + 'return time, states\n\n')

        elif method_name == 'retrieve_results' and self.name_states is not None:
            if self.model_type == 'ODE':
                assign_outputs = [
                    ' '*8 + 'outputs_split = np.split(states, np.acum_len, axis=1)\n',
                    ' '*8 + 'model_outputs = dict(zip(self.name_states, outputs_split))\n',
                    ]
            elif self.model_type == 'DAE':
                assign_outputs = [
                    ' '*8 + 'outputs_split = np.split(states, np.acum_len, axis=1)\n',
                    ' '*8 + 'model_outputs = dict(zip(self.name_states, outputs_split))\n',
                    ]
            elif self.model_type == 'PDE':
                assign_outputs = [
                    ' '*8 + 'model_outputs = reorder_pde_outputs(states, self.num_nodes, self.len_states)']

            open_object.writelines(assign_outputs)

        elif method_name != 'Phases':
            open_object.write(' '*8 + 'return\n\n')

    def CreatePharmaPyTemplate(self):
        op = open(self.file_name, 'w')

        if self.model_type == 'DAE':
            self.solver = 'IDA'
            self.problem = 'Implicit_Problem'
        else:
            self.solver = 'CVode'
            self.problem = 'Explicit_Problem'

        if self.model_type == 'PDE':
            pharmapy_modules = [
                'from PharmaPy.Commons import reorder_pde_outputs\n\n\n']
        else:
            pharmapy_modules = []

        packages = [
            'import numpy as np\n',
            'from assimulo.problem import %s\n' % self.problem,
            'from PharmaPy.Phases import classify_phases\n',
            'from assimulo.solvers import %s\n\n' % self.solver,
            '\n']

        packages += pharmapy_modules

        op.writelines(packages)

        if self.model_type == 'PDE':
            args_init = ['num_nodes']
        elif self.model_type == 'ODE':
            args_init = []
        elif self.model_type == 'DAE':
            args_init = []

        if self.has_stages:
            args_init.append('num_stages')

        displayed_args_init = [' '*8 + 'self.{} = {}\n'.format(arg, arg)
                               for arg in args_init]

        args_init = ', '.join(args_init)

        op.write('class {}:\n'.format(self.class_name))

        init_lines = [' ' * 4 + 'def __init__(self, {}):\n\n'.format(args_init),
                      ' ' * 8 + 'self.nomenclature()\n',
                      ' ' * 8 + 'self._Phases = None\n']

        if self.oper_mode != 'batch':
            init_lines.append(' ' * 8 + 'self._Inlet = None\n')

        op.writelines(init_lines)
        op.writelines(displayed_args_init)

        op.write(' ' * 8 + 'return\n\n')

        # Nomenclature snippet
        name_states = list(self.name_states)
        nomenclature = [
            ' '*4 + 'def nomenclature(self):\n',
            ' '*8 + 'self.name_states = {}\n\n'.format(name_states)]

        op.writelines(nomenclature)

        # Inlets (if any)
        if self.oper_mode != 'batch':
            inlet_prop = [' ' * 4 + '@property\n',
                          ' ' * 4 + 'def Inlet(self):\n',
                          ' ' * 8 + 'return self._Inlet\n\n']

            inlet_set = [' ' * 4 + '@Inlet.setter\n',
                         ' ' * 4 + 'def Inlet(self, inlet):\n',
                         ' ' * 8 + 'self._Inlet = inlet\n\n']

            op.writelines(inlet_prop)
            op.writelines(inlet_set)

        for method, args in self.method_arguments.items():
            self.__write_method(op, method, args)

        op.close()


if __name__ == '__main__':
    name_file = 'DynamicExtraction.py'
    name_class = 'DynamicExtractor'
    states = ['mol_i', 'x_liq', 'y_liq', 'holdup_R', 'holdup_E',
              'R_flow', 'E_flow', 'u_int', 'temp']

    meta_object = MetaModelingClass(name_file, name_class, name_states=states,
                                    model_type='DAE')
    meta_object.CreatePharmaPyTemplate()

    # name_file = 'test_pde.py'
    # name_class = 'DistillationColumn'
    # states = ['x_liq', 'liq_holdup', 'temp', 'vap_flows']

    # meta_object = MetaModelingClass(name_file, name_class, name_states=states,
    #                                 model_type='PDE', oper_mode='continuous')
    # meta_object.CreatePharmaPyTemplate()
