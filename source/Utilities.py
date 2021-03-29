#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:52:55 2020

@author: dcasasor
"""


class CoolingWater:
    def __init__(self, vol_flow=0, mass_flow=0, temp_in=298.15):
        self.rho = 1000  # kg/m**3
        self.cp = 4180  # J/kg/K

        if vol_flow > 0:
            self.vol_flow = vol_flow
            self.mass_flow = vol_flow * self.rho  # kg/s
        elif mass_flow > 0:
            self.vol_flow = mass_flow / self.rho  # m**3/s
            self.mass_flow = mass_flow
        self.temp_in = temp_in

        # Outputs
        self.temp_out = None
