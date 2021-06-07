# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:37:19 2021

@author: huri
"""

import numpy as np
from scipy.interpolate import CubicSpline


def define_initial_state(state, z_after, z_before=None, indexed_state=False):
       
    if indexed_state:  # e.g. concentration
        if state.ndim == 1:
            num_z = len(z_after)
            state_interp = np.tile(state, (num_z, 1))
        else:
            interp_obj = CubicSpline(z_before, state) 
            state_interp = interp_obj(z_after)
            
    else:  # e.g. saturation
        if isinstance(state, float):
            state_interp = state * np.ones_like(z_after)
        else:
            interp_obj = CubicSpline(z_before, state)
            state_interp = interp_obj(z_after)
    
    return state_interp