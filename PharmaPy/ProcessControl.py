# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:06:16 2021

@author: dcasasor
"""


class DynamicInput:
    def __init__(self):
        self.controls = {}
        self.args_control = {}

        # Attributes assigned from UO instance
        self.parent_instance = None

    def add_control(self, variable_name, function, args_control=None):
        self.controls[variable_name] = function
        if args_control is None:
            args_control = ()

        self.args_control[variable_name] = args_control

    def evaluate_inputs(self, time):
        controls_out = {}

        for key, fun in self.controls.items():
            args = self.args_control[key]
            controls_out[key] = fun(time, *args)

        for name in self.parent_instance.controllable:
            if name not in controls_out.keys():
                controls_out[name] = getattr(self.parent_instance, name)

        return controls_out