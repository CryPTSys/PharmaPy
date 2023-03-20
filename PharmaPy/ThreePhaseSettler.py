# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:51:16 2023

@author: vsundark
"""

import numpy as np
from assimulo.problem import Implicit_Problem
from PharmaPy.Phases import classify_phases
from PharmaPy.Streams import VaporStream
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Commons import unpack_discretized
from PharmaPy.Streams import LiquidStream
from PharmaPy.Results import DynamicResult
from PharmaPy.Plotting import plot_distrib

from assimulo.solvers import IDA

import scipy.optimize
import scipy.sparse

# from itertools import cycle
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

class ThreePhaseSettler:
    def __init__(self, d_impeller, d_vessel, height_toplayer, 
                 height_botlayer, agit_rate=1000/60, **other_phys_props):

        self.d_impeller = d_impeller
        self.d_vessel = d_vessel
        self.height_toplayer = height_toplayer
        self.height_botlayer = height_botlayer
        self.agit_rate = agit_rate
        
        self.holdup_top = np.pi*d_vessel**2/4*height_toplayer
        self.holdup_bot = np.pi*d_vessel**2/4*height_botlayer
        self.other_phys_props = other_phys_props
        self.nomenclature()
        return
    
    def nomenclature(self):
        self.name_states = ['mass_frac']
        self.states_di = {'mass_frac': {'dim': 1, 'units': 'kg API/kg solvent'}}
        self.states_dict = {'Inlet': self.states_di}
        self.fstates_di = {}
        
    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if not isinstance(phases, (list, tuple)):
            phases = [phases]

        self._Phases = phases
        classify_phases(self)
    
    @property
    def Inlet_feed(self):
        return self._Inlet_feed
    @Inlet_feed.setter
    def Inlet_feed(self, inlet_feed):
        self._Inlet_feed = inlet_feed
        self.name_species = self.Inlet_feed.name_species
        self.index = self.name_species.index('lomustine_crystals')
        
    @property
    def Inlet_bot_phase(self):
        return self._Inlet_bot_phase
    @Inlet_bot_phase.setter
    def Inlet_bot_phase(self, inlet_bot_phase):
        self._Inlet_bot_phase = inlet_bot_phase
    
    def get_input_feed(self, time):
        inputs = get_inputs_new(time, self.Inlet_feed, self.states_dict)
        return inputs
    
    def get_input_bot_phase(self, time):
        inputs = get_inputs_new(time, self.Inlet_bot_phase, self.states_dict)
        return inputs
    
    def unit_model(self, time, states, d_states):
        material = self.material_balances(time, states)
        material = material - d_states
        return material
    
    def design_space_check(self,):
        d_particle = self.other_phys_props['d_particle']
        viscosity_top = self.other_phys_props['top_viscosity']
        density_top = self.other_phys_props['top_density_fluid']
        density_particle = self.other_phys_props['density_particle']
        flowrate_feed = self.Inlet_feed.vol_flow
        gamma_top = self.other_phys_props['top_gamma']
        g = 9.81 #m/s2
        
        # 1st criterion
        vt_top = g*d_particle**2*(density_particle-density_top)/(18*viscosity_top)
        tau_top = self.holdup_top/flowrate_feed
        vt_crit = 10* self.height_toplayer/tau_top
        crit1 = vt_top > vt_crit
        
        viscosity_bot = self.other_phys_props['bot_viscosity']
        density_bot = self.other_phys_props['bot_density_fluid']
        flowrate_bot = self.Inlet_bot_phase.vol_flow
        gamma_bot = self.other_phys_props['bot_gamma']
        gamma_int = np.abs(gamma_bot-gamma_top)
        
        d_impeller = self.d_impeller
        agit_rate = self.agit_rate
        
        # 2nd criterion
        Bo = (density_particle-density_bot)*g*d_particle**2/gamma_int
        Ea = np.pi*d_particle**2/4*gamma_int
        Ek = 0.5*(density_particle*np.pi*d_particle**3/6)*(d_impeller/2*agit_rate/(2*np.pi))
        crit2 = Bo>1 or Ek>Ea
        
        def N_cr_cal(N):
            N = np.abs(N)
            if len(N):
                N=N[-1]
            Re = density_bot*N*d_impeller**2/viscosity_bot
            Fr = N**2*d_impeller/g
            zeta = 1
            return d_impeller - d_impeller*Fr*(12.9*(1-Re**-0.11*zeta**0.17)-4.27)
        
        # 3rd criterion
        N_cr = scipy.optimize.fsolve(N_cr_cal, x0=agit_rate)
        N_js = (234*(d_impeller)**(-2/3)*(d_particle)**(1/3)
                *((density_particle-density_bot)/density_bot)**(2/3)
                *(viscosity_bot/density_bot)**(-1/9))
        crit3 = agit_rate<N_cr and agit_rate>N_js
        
        # Test
        if crit1 and crit2 and crit3:
            print('Operating point is in design space')
        else:
            print('Operating point outside design space')
        
        return
    
    def material_balances(self, time, mass_frac):
        density_top = self.other_phys_props['top_density_fluid']
        flowrate_feed = self.Inlet_feed.vol_flow
        density_bot = self.other_phys_props['bot_density_fluid']
        flowrate_bot = self.Inlet_bot_phase.vol_flow
        
        input_feed = self.get_input_feed(time)['Inlet']
        input_bot_phase = self.get_input_bot_phase(time)['Inlet']
        mass_frac_feed = input_feed['mass_frac'][self.index]

        d_mass_frac_dt = ((1/(self.holdup_bot*density_bot))
                          *(mass_frac_feed*flowrate_feed*density_top
                            -mass_frac*flowrate_bot*density_bot))
        
        return d_mass_frac_dt

    def energy_balances(self, time, temp, mole_frac):
        pass
        return
    
    def solve_unit(self, runtime=None, t0=0):
        #Check if operating point is in design space
        self.design_space_check()

        init_states = self.Inlet_bot_phase.mass_frac[-1].copy()
        init_derivative = self.material_balances(time=0, mass_frac=init_states)

        problem = Implicit_Problem(self.unit_model, init_states, init_derivative, t0)
        solver = IDA(problem)
        time, states, d_states = solver.simulate(runtime)
        self.retrieve_results(time, states)
        
        return time, states, d_states
    
    def retrieve_results(self, time, states):
        time = np.asarray(time)
        self.timeProf = time

        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        dp = {'time': time, 'mass_frac': states}
        self.result = DynamicResult(di_states=self.states_di, di_fstates=None,**dp)
        self.outputs = dp
        
        # Outlet stream
        path = self.Inlet_bot_phase.path_data
        mass_frac = self.Inlet_bot_phase.mass_frac
        mass_frac[-1] = dp['mass_frac'][-1][-1]
        mass_frac[-2] = mass_frac[-2] - dp['mass_frac'][-1][-1]
        temp=self.Inlet_bot_phase.temp
        self.OutletBottom = LiquidStream(
            path, mass_frac=mass_frac,  vol_flow=self.Inlet_bot_phase.vol_flow)
        self.Outlet = self.OutletBottom