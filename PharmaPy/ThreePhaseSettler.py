# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:51:16 2023

@author: vsundark
"""

import numpy as np
from assimulo.problem import Implicit_Problem
from PharmaPy.Phases import classify_phases
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Streams import LiquidStream, SolidStream
from PharmaPy.MixedPhases import Slurry, SlurryStream
from PharmaPy.Results import DynamicResult
from PharmaPy.Plotting import plot_distrib

from assimulo.solvers import IDA

import scipy.optimize
import scipy.sparse

# from itertools import cycle
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

class ThreePhaseSettler:
    def __init__(self, d_impeller, d_vessel,
                 target_compound, agit_rate=1000/60):

        self.d_impeller = d_impeller
        self.d_vessel = d_vessel
        self.target_compound = target_compound
        self.agit_rate = agit_rate

        self.outputs = None
        self.oper_mode = 'Continuous'

    def nomenclature(self):
        self.name_states = ['solid_conc']
        self.states_di = {'solid_conc':
                          {'dim': 1, 'units': 'kg API/kg solvent'}}
        self.states_dict = {'Inlet': self.states_di}
        self.fstates_di = {}

        self.names_states_in = ['solid_conc']

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if not isinstance(phases, (list, tuple)):
            phases = [phases]

        self._Phases = phases
        classify_phases(self)

        if self.Liquid_1.getDensity() > self.Liquid_2.getDensity():
            self.BotLiq = self.Liquid_1
            self.TopLiq = self.Liquid_2
        else:
            self.BotLiq = self.Liquid_2
            self.TopLiq = self.Liquid_1
        path = self.Inlet_bot_phase.path_data
        self.TopPhase = Slurry(path)
        self.TopPhase.Phases = [self.TopLiq, self.Solid_1]
        self.BotPhase = Slurry(path)
        self.BotPhase.Phases = [self.BotLiq, self.Solid_2]
        self.holdup_top = self.TopLiq.vol
        self.holdup_bot = self.BotLiq.vol

        self.nomenclature()

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, Inlet):
        self._Inlet = Inlet
        self.name_species = self.Inlet.Liquid_1.name_species
        self.index = self.name_species.index(self.target_compound)

    @property
    def Inlet_bot_phase(self):
        return self._Inlet_bot_phase

    @Inlet_bot_phase.setter
    def Inlet_bot_phase(self, inlet_bot_phase):
        self._Inlet_bot_phase = inlet_bot_phase
        
    def flatten_states(self):
        pass

    def get_input_feed(self, time):
        inputs = get_inputs_new(time, self.Inlet, self.states_dict)
        return inputs

    def get_input_bot_phase(self, time):
        inputs = get_inputs_new(time, self.Inlet_bot_phase, self.states_dict)
        return inputs

    def unit_model(self, time, states, d_states):
        material = self.material_balances(time, states)
        material = material - d_states
        return material

    def design_space_check(self,):

        # Calculate mean diameter
        distrib = self.Inlet.Solid_1.distrib
        x_distrib = self.Inlet.Solid_1.x_distrib
        perc_vol_basis = np.percentile(distrib*x_distrib**3*1e-9, 50,
                                       method='closest_observation') #distrib**3 to get volume basis, 1e-9 to convert to m3 and reduce number magnitude
        D50_index = np.where(distrib*x_distrib**3*1e-9==perc_vol_basis)[0][0]
        D50 = x_distrib[D50_index]

        d_particle = D50*1e-6

        # d_particle = np.dot(self.Inlet.Solid_1.distrib/sum(self.Inlet.Solid_1.distrib),
        #                     self.Inlet.Solid_1.x_distrib/1e6)
        viscosity_top = self.TopPhase.Liquid_1.getViscosity()
        density_top = self.TopPhase.Liquid_1.getDensity()
        density_particle = self.TopPhase.Solid_1.getDensity()
        flowrate_feed = self.Inlet.Liquid_1.vol_flow
        gamma_top = self.TopPhase.Liquid_1.getSurfTension()
        g = 9.81  # m/s2

        # 1st criterion
        vt_top = g*d_particle**2*(density_particle-density_top)/(18*viscosity_top)
        tau_top = self.holdup_top/flowrate_feed
        height_toplayer = self.holdup_top/(np.pi*self.d_vessel**2/4)
        vt_crit = 10* height_toplayer/tau_top
        crit1 = vt_top > vt_crit

        viscosity_bot = self.BotPhase.Liquid_1.getViscosity()
        density_bot = self.TopPhase.Liquid_1.getDensity()
        flowrate_bot = self.Inlet_bot_phase.vol_flow
        gamma_bot = self.BotPhase.Liquid_1.getSurfTension()
        gamma_int = np.abs(gamma_bot-gamma_top)

        d_impeller = self.d_impeller
        agit_rate = self.agit_rate

        # 2nd criterion
        Bo = (density_particle-density_bot)*g*d_particle**2/gamma_int
        Ea = np.pi*d_particle**2/4*gamma_int
        Ek = 0.5*(density_particle*np.pi*d_particle**3/6)*(d_impeller/2*(agit_rate/2*np.pi))
        crit2 = Bo>1 or Ek>Ea

        def N_cr_cal(N):
            N = np.abs(N)
            if len(N):
                N=N[-1]
            Re = density_bot*N*d_impeller**2/viscosity_bot
            Fr = N**2*d_impeller/g
            zeta = 0.85
            return zeta - d_impeller*Fr*(12.9*(1-Re**-0.11*zeta**0.17)-4.27)

        # 3rd criterion
        N_cr = scipy.optimize.fsolve(N_cr_cal, x0=agit_rate)[0]
        N_js = (234*(d_impeller)**(-2/3)*(d_particle)**(1/3)
                *((density_particle-density_bot)/density_bot)**(2/3)
                *(viscosity_bot*1000/(density_bot/1000))**(-1/9)/60)
        crit3 = agit_rate<N_cr and agit_rate>N_js

        # Test
        if crit1 and crit2 and crit3:
            print('TPS: operating point is in design space')
        else:
            print('TPS: operating point is not in design space')

        return

    def material_balances(self, time, solid_conc):
        density_top = self.TopPhase.Liquid_1.getDensity()
        flowrate_feed = self.Inlet.Liquid_1.vol_flow
        density_bot = self.BotPhase.Liquid_1.getDensity()
        flowrate_bot = self.Inlet_bot_phase.vol_flow

        input_feed = self.get_input_feed(time)['Inlet']
        solid_conc_feed = input_feed['solid_conc']
        if isinstance(solid_conc_feed, (list, np.ndarray)):
            solid_conc_feed = solid_conc_feed[solid_conc_feed!=0][0]

        d_solid_conc_dt = ((1/(self.holdup_bot * density_bot))
                          *(solid_conc_feed*flowrate_feed*density_top
                            -solid_conc*flowrate_bot*density_bot))

        return d_solid_conc_dt

    def energy_balances(self, time, temp, mole_frac):
        pass
        return

    def solve_unit(self, runtime=None, t0=0, verbose=True):
        # Check if operating point is in design space
        self.design_space_check()

        init_states = (self.BotPhase.Solid_1.mass 
                       / (self.BotPhase.Liquid_1.mass + self.BotPhase.Solid_1.mass))
        init_derivative = self.material_balances(time=0,
                                                 solid_conc=init_states)

        problem = Implicit_Problem(self.unit_model, init_states,
                                   init_derivative, t0)
        solver = IDA(problem)

        if not verbose:
            solver.verbosity = 50

        time, states, d_states = solver.simulate(runtime)
        self.retrieve_results(time, states)
        return time, states, d_states

    def retrieve_results(self, time, states):
        time = np.asarray(time)
        self.timeProf = time

        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        dp = {'time': time, 'solid_conc': states}
        self.result = DynamicResult(di_states=self.states_di, di_fstates=None,**dp)
        self.outputs = dp

        # Outlet stream
        path = self.Inlet_bot_phase.path_data
        solid_conc = self.result.solid_conc[-1][-1]
        solid_mass_flow = solid_conc*self.Inlet_bot_phase.vol_flow*self.Inlet_bot_phase.getDensity()

        OutletLiq = LiquidStream(
             path, mass_frac=self.Inlet_bot_phase.mass_frac,  vol_flow=self.Inlet_bot_phase.vol_flow)
        OutletSolid = SolidStream(
             path, mass_flow=solid_mass_flow,  mass_frac=self.Inlet.Solid_1.mass_frac,
             temp=self.Inlet.Solid_1.temp, distrib=self.Inlet.Solid_1.distrib, x_distrib=self.Inlet.Solid_1.x_distrib)
        self.Outlet = SlurryStream(path)
        self.Outlet.Phases = [OutletLiq, OutletSolid]
