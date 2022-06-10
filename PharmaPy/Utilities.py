#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:52:55 2020

@author: dcasasor
"""

from PharmaPy.Connections import get_inputs_new


class CoolingWater:
    def __init__(self, vol_flow=None, mass_flow=None, temp_in=298.15,
                 h_conv=1000):

        self.rho = 1000  # kg/m**3
        self.cp = 4180  # J/kg/K
        self.h_conv = h_conv

        if vol_flow is None and mass_flow is None:
            raise RuntimeError("Both 'vol_flow' and 'mass_flow' are None. "
                               "Specify one of them.")

        self.updateObject(vol_flow, mass_flow, temp_in)

        self.controllable = ('temp_in', 'vol_flow', 'mass_flow')

        # Outputs
        self.temp_out = None
        self.y_upstream = None

        self._DynamicInlet = None

    @property
    def DynamicInlet(self):
        return self._DynamicInlet

    @DynamicInlet.setter
    def DynamicInlet(self, dynamic_object):
        dynamic_object.controllable = self.controllable
        dynamic_object.parent_instance = self

        self._DynamicInlet = dynamic_object

    def updateObject(self, vol_flow=None, mass_flow=None, temp_in=None):
        if vol_flow is not None:
            self.vol_flow = vol_flow
            self.mass_flow = vol_flow * self.rho
        elif mass_flow is not None:
            self.mass_flow = mass_flow
            self.vol_flow = mass_flow / self.rho

        if temp_in is not None:
            self.temp_in = temp_in

    def evaluate_inputs(self, time):
        if self.DynamicInlet is None:
            inputs = {}
            for attr in self.controllable:
                inputs[attr] = getattr(self, attr)

        else:
            inputs = self.DynamicInlet.evaluate_inputs(time)

        return inputs

    def get_inputs(self, time):
        di = {'Inlet': {'vol_flow': 1, 'temp_in': 1, 'mass_flow': 1}}
        inputs = get_inputs_new(time, self, di)['Inlet']

        return inputs
