#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:52:55 2020

@author: dcasasor
"""


class CoolingWater:
    def __init__(self, vol_flow=None, mass_flow=None, temp_in=298.15,
                 controls=None, args_control=None, h_conv=1000):

        self.rho = 1000  # kg/m**3
        self.cp = 4180  # J/kg/K
        self.h_conv = h_conv

        if vol_flow is None and mass_flow is None:
            raise RuntimeError("Both 'vol_flow' and 'mass_flow' are None. "
                               "Specify one of them.")

        self.updateObject(vol_flow, mass_flow, temp_in)

        # Controls
        if controls is None:
            controls = {}
        else:
            if args_control is None:
                args_control = {key: () for key in controls.keys()}

            update_dict = {}
            for key, fun in controls.items():
                update_dict[key] = fun(0, *args_control[key])

            self.updateObject(**update_dict)

        self.controls = controls
        self.args_control = args_control
        self.controllable = ('temp_in', 'vol_flow', 'mass_flow')

        # Outputs
        self.temp_out = None

    def updateObject(self, vol_flow=None, mass_flow=None, temp_in=None):
        if vol_flow is not None:
            self.vol_flow = vol_flow
            self.mass_flow = vol_flow * self.rho
        elif mass_flow is not None:
            self.mass_flow = mass_flow
            self.vol_flow = mass_flow / self.rho

        if temp_in is not None:
            self.temp_in = temp_in

    def evaluate_controls(self, time):
        controls_out = {}

        if len(self.controls) > 0:
            for key, fun in self.controls.items():
                args = self.args_control[key]
                controls_out[key] = fun(time, *args)

        for name in self.controllable:
            if name not in controls_out.keys():
                controls_out[name] = getattr(self, name)

        return controls_out
