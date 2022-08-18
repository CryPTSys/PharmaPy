# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:06:16 2021

@author: dcasasor
"""


def analyze_controls(di):

    controls = {}

    for key, val in di.items():
        if isinstance(val, dict) and key != 'kwargs':
            if 'fun' not in val:
                raise KeyError("'%s' dictionary must have a 'fun' field")
            elif not callable(val['fun']):
                raise TypeError(
                    "Object passed to the 'fun' field must be a "
                    "callable with signature fun(time, *args, **kwargs)")

            out = val
        else:
            if not callable(val):
                raise TypeError(
                    "Object passed to the 'fun' field must be a "
                    "callable with signature fun(time, *args, **kwargs)")

            out = {'fun': val}

        if 'args' not in out:
            out['args'] = ()

        if 'kwargs' not in out:
            out['kwargs'] = {}

        controls[key] = out

    return controls


class DynamicInput:
    def __init__(self):
        self.controls = {}
        self.args_control = {}

        # Attributes assigned from UO instance
        self.parent_instance = None

    def add_variable(self, variable_name, function, args_control=None):
        self.controls[variable_name] = function
        if args_control is None:
            args_control = ()

        self.args_control[variable_name] = args_control

    def evaluate_inputs(self, time):
        controls_out = {}

        for key, fun in self.controls.items():
            args = self.args_control[key]
            controls_out[key] = fun(time, *args)

        return controls_out
