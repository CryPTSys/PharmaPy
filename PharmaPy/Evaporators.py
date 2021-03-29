# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:37:29 2020

@author: dcasasor
"""

from autograd import numpy as np
import numpy as np
from autograd import jacobian as jacauto
from PharmaPy.Commons import mid_fn
from PharmaPy.Streams import LiquidStream, VaporStream
from PharmaPy.Phases import LiquidPhase, VaporPhase
from PharmaPy.NameAnalysis import get_dict_states

from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA

from pathlib import Path

gas_ct = 8.314
eps = np.finfo(float).eps


class IsothermalFlash:
    def __init__(self, Inlet=None, temp_drum=None, pres_drum=None,
                 gamma_method='ideal'):

        self.temp = temp_drum
        self.pres = pres_drum
        self._Inlet = Inlet

        self.gamma_method = gamma_method

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, instance):
        self._Inlet = instance

        if self.temp is None:
            self.temp = self._Inlet.temp

        if self.pres is None:
            self.pres = self._Inlet.pres

        # self.pres = self._Inlet.pres
        self.num_comp = self._Inlet.num_species

        # if self.k_func is None:
        self.k_func = self.Inlet.getKeqVLE

        if self.Inlet.__module__ == 'PharmaPy.Streams':
            self.in_flow = self.Inlet.mole_flow
        else:
            self.in_flow = self.Inlet.moles

    def material_balance(self, liq, vap, x_i, y_i, z_i):

        k_i = self.k_func(self.temp, self.pres, gamma_model=self.gamma_method)

        global_bce = 1 - liq - vap
        comp_bces = z_i - liq*x_i - vap*y_i
        equilibria = y_i - k_i * x_i

        diff_frac = np.sum(x_i - y_i)
        args_mid = np.array([vap, diff_frac, vap - 1])
        vap_flow = mid_fn(args_mid)

        balance = np.concatenate(
            (np.array([global_bce]),
             comp_bces, equilibria,
             np.array([vap_flow]))
            )

        return balance

    def energy_balance(self, liq, vap, x_i, y_i, z_i):
        liq_flow = liq * self.in_flow  # mol/s
        vap_flow = vap * self.in_flow  # mol/s

        tref = self.temp

        h_liq = 0
        h_vap = self.VaporOut.getEnthalpy(self.temp, temp_ref=tref,
                                          mole_frac=y_i, basis='mole')

        h_in = self.Inlet.getEnthalpy(temp_ref=tref, basis='mole')

        heat_duty = liq_flow * h_liq + vap_flow * h_vap - self.in_flow * h_in

        return heat_duty  # W

    def unit_model(self, states, material=True):
        liq_pr = states[0]
        vap_pr = states[-1]

        fracs = states[1:-1]
        x_i = fracs[:self.num_comp]
        y_i = fracs[self.num_comp:]
        z_i = self.Inlet.mole_frac

        if material:
            balance = self.material_balance(liq_pr, vap_pr, x_i, y_i, z_i)
        else:
            balance = self.energy_balance(liq_pr, vap_pr, x_i, y_i, z_i)

        return balance

    def solve_unit(self):

        # Set seeds
        l_seed = 0.5
        v_seed = 0.5

        x_seed = np.ones(self.num_comp) * 1 / self.num_comp
        y_seed = x_seed

        seed = np.concatenate(([l_seed], x_seed, y_seed, [v_seed]))

        # # Set jacobian
        # jac_eqns = jacauto(self.unit_model)

        # Solve nonlinear system
        solution = fsolve(self.unit_model, seed, full_output=False)
                          # fprime=jac_eqns,

        # Retrieve solution
        liq_flash = solution[0] * self.in_flow
        vap_flash = solution[-1] * self.in_flow

        fracs = solution[1:2*self.num_comp + 1]
        x_flash = fracs[:self.num_comp]
        y_flash = fracs[self.num_comp:]

        path = self.Inlet.path_data

        # Store in objects
        if self.Inlet.__module__ == 'PharmaPy.Streams':
            self.LiquidOut = LiquidStream(path, mole_frac=x_flash,
                                          mole_flow=liq_flash,
                                          temp=self.temp, pres=self.pres)
            self.VaporOut = VaporStream(path, mole_frac=y_flash,
                                        mole_flow=vap_flash,
                                        temp=self.temp, pres=self.pres)
        else:
            self.LiquidOut = LiquidPhase(path, mole_frac=x_flash,
                                         moles=liq_flash,
                                         temp=self.temp, pres=self.pres)
            self.VaporOut = VaporPhase(path, mole_frac=y_flash,
                                       moles=vap_flash,
                                       temp=self.temp, pres=self.pres)

        heat_bce = self.unit_model(solution, material=False)

        return solution, heat_bce


class AdiabaticFlash:
    def __init__(self, pres_drum, div_factor=1e3, gamma_method='ideal'):
        self.pres = pres_drum
        self.div_factor = div_factor

        self._Inlet = None

        self.gamma_method = gamma_method

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, instance):
        self._Inlet = instance
        self.num_comp = self._Inlet.num_species

        path = self.Inlet.path_data

        # self.LiquidOut = LiquidStream(path, pres=self.pres)
        self.VaporOut = VaporStream(path, pres=self.pres)

        # if self.k_func is None:
        self.k_func = self.Inlet.getKeqVLE

        if self.Inlet.__module__ == 'PharmaPy.Streams':
            self.in_flow = self.Inlet.mole_flow
        else:
            self.in_flow = self.Inlet.moles

    def material_balances(self, liq, vap, x_i, y_i, z_i, temp):

        k_i = self.Inlet.getKeqVLE(temp, self.pres, x_i, y_i,
                                   gamma_model=self.gamma_method)

        # Material balances
        global_bce = 1 - liq - vap
        comp_bces = z_i - liq*x_i - vap*y_i
        equilibria = y_i - k_i * x_i

        diff_frac = np.sum(x_i - y_i)
        args_mid = np.array([vap, diff_frac, vap - 1])
        vap_flow = mid_fn(args_mid)

        material = np.concatenate((
            np.array([global_bce]),
            comp_bces, equilibria,
            np.array([vap_flow]))
            )

        return material

    def energy_balance(self, liq, vap, x_i, y_i, z_i, temp):

        self.VaporOut.mole_frac = y_i

        # Energy balance
        h_liq = 0  # reference state
        h_vap = self.VaporOut.getEnthalpy(temp, temp_ref=temp, basis='mole')
        h_feed = self.Inlet.getEnthalpy(temp_ref=temp, basis='mole')

        energy_bce = liq * h_liq + vap * h_vap - h_feed
        energy_bce = np.atleast_1d(energy_bce / self.div_factor)

        return energy_bce

    def unit_model(self, states):
        liq = states[0]
        vap = states[-2]

        fracs = states[1:-2]
        x_i = fracs[:self.num_comp]
        y_i = fracs[self.num_comp:]
        z_i = self.Inlet.mole_frac

        temp = states[-1]

        material_bces = self.material_balances(liq, vap, x_i, y_i, z_i, temp)
        energy_bce = self.energy_balance(liq, vap, x_i, y_i, z_i, temp)

        balances = np.concatenate((material_bces, energy_bce))

        return balances

    def solve_unit(self):
        l_seed = 0.5
        x_seed = np.ones(self.num_comp) * 1 / self.num_comp
        y_seed = x_seed
        v_seed = 0.5

        temp_seed = self.Inlet.temp

        seed = np.concatenate(([l_seed], x_seed, y_seed, [v_seed, temp_seed]))

        # jac_eqns = jacauto(self.unit_model)
        solution = fsolve(self.unit_model, seed)
                          # fprime=jac_eqns)

        # Retrieve solution
        liq_flash = solution[0] * self.in_flow
        vap_flash = solution[-2] * self.in_flow
        temp = solution[-1]

        fracs = solution[1:2*self.num_comp + 1]
        x_flash = fracs[:self.num_comp]
        y_flash = fracs[self.num_comp:]

        path = self.Inlet.path_data

        # Store in objects
        if self.Inlet.__module__ == 'PharmaPy.Streams':
            self.LiquidOut = LiquidStream(path, mole_frac=x_flash,
                                          mole_flow=liq_flash,
                                          temp=temp, pres=self.pres)
            self.VaporOut = VaporStream(path, mole_frac=y_flash,
                                        mole_flow=vap_flash,
                                        temp=temp, pres=self.pres)
        else:
            self.LiquidOut = LiquidPhase(path, mole_frac=x_flash,
                                         moles=liq_flash,
                                         temp=temp, pres=self.pres)
            self.VaporOut = VaporPhase(path, mole_frac=y_flash,
                                       moles=vap_flash,
                                       temp=temp, pres=self.pres)

        return solution


class Evaporator:
    def __init__(self, vol_drum, phase=None, inlet=None,
                 pres=101325, diam_out=2.54e-2,
                 ht_coeff=1000, temp_ht=298.15,
                 activity_model='ideal'):

        self._Inlet = inlet
        self._Phases = phase

        self.vol_tot = vol_drum
        self.area_out = np.pi / 4 * diam_out**2

        self.pres = pres

        # Jacobians
        self.jac_states = jacauto(self.unit_model, 1)
        self.jac_sdot = jacauto(self.unit_model, 2)

        # Heat transfer
        self.ht_coeff = ht_coeff
        self.temp_ht = temp_ht

        self.oper_mode = 'Batch'  # If inlet setter, then Semibatch

        self.is_continuous = False

        self.timeProf = []
        self.tempProf = []
        self.presProf = []

        self.xliqProf = []
        self.yvapProf = []

        self.activity_model = activity_model

        self.nomenclature()

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phase):
        path_comp = phase.path_data
        path_inert = '/data/evaporator/props_nitrogen.json'
        path_inert = str(Path(__file__).parents[2]) + path_inert
        paths = [path_comp, path_inert]
        self.paths = paths

        self.LiqPhase = phase

        self.num_species = len(self.LiqPhase.mole_frac)
        # self.name_species = self.Inlet.name_species
        self.name_species = phase.name_species

        self.VaporOut = VaporStream(path_comp, pres=self.pres)

        self._Phases = phase


    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):  # Create an inlet with additional species (N2)
        path_comp = inlet.path_data
        path_inert = '/data/evaporator/props_nitrogen.json'
        path_inert = str(Path(__file__).parents[2]) + path_inert

        paths = [path_comp, path_inert]

        fields = ['temp', 'pres', 'mole_flow', 'mole_frac']
        inlet_dict = {key: inlet.__dict__.get(key) for key in fields}
        inlet_dict['mole_frac'] = np.append(inlet_dict['mole_frac'], 0)

        inlet_inert = LiquidStream(paths, **inlet_dict)
        self._Inlet = inlet_inert

        self.oper_mode = 'Semibatch'
        # self.paths = paths

    def nomenclature(self):
        self.names_states_in = ['mole_frac', 'mole_flow', 'temp']
        self.names_states_out = ['mole_frac', 'moles', 'temp']

    def get_inputs(self, time):
        if self.Inlet is None:
            input_dict = {'mole_flow': 0,
                          'mole_frac': np.zeros(self.num_species + 1),
                          'temp': 298.15}
        elif hasattr(self.Inlet, 'y_inlet'):
            pass  # TODO: Interpolate
        else:
            input_dict = {'mole_flow': self.Inlet.mole_flow,
                          'mole_frac': self.Inlet.mole_frac,
                          'temp': self.Inlet.temp}

        return input_dict

    def material_balances(self, dmoli_dt, moles_i, x_i, y_i,
                          mol_liq, mol_vap, pres, temp, u_inputs):

        input_flow = u_inputs['mole_flow']
        input_fracs = u_inputs['mole_frac']

        mw_vap = self.VapEvap.mw_av

        dens_gas = pres / gas_ct / temp * (mw_vap / 1000)  # kg/m**3
        flow_vap = self.area_out / mw_vap * 1000 *\
            np.sqrt(2*dens_gas * np.max([eps, pres - self.pres]))  # mol/s

        # Differential eqns
        diff_i = input_flow * input_fracs - flow_vap * y_i - dmoli_dt  # bce - dMi_dt

        # Algebraic eqns
        component_bce = mol_liq * x_i + mol_vap * y_i - moles_i
        global_bce = mol_liq + mol_vap - sum(moles_i)

        rho_liq = self.LiqEvap.getDensity(mole_frac=x_i, temp=temp,
                                          basis='mole')

        vol_liq = mol_liq / rho_liq / 1000  # m**3
        vol_vap = mol_vap * gas_ct * temp / pres
        vol_eqn = vol_liq + vol_vap - self.vol_tot

        k_i = self.LiqEvap.getKeqVLE(temp, pres, x_i, self.activity_model)
        equilibria = y_i - k_i * x_i

        p_sat = self.LiqEvap.AntoineEquation(temp=temp)
        pres_eqn = np.dot(x_i, p_sat) - pres

        alg_balances = np.concatenate((component_bce,  # x_i
                                       equilibria,  # y_i
                                       np.array([global_bce]),  # M_L
                                       np.array([vol_eqn]),  # M_V
                                       np.array([pres_eqn])  # P
                                       )
                                      )

        return diff_i, alg_balances, flow_vap, vol_liq

    def energy_balances(self, du_dt, u_int, temp, x_i, y_i, mol_liq, mol_vap,
                        vol_liq, flow_vap, pres, u_inputs):

        input_flow = u_inputs['mole_flow']
        input_fracs = u_inputs['mole_frac']
        input_temp = u_inputs['temp']

        # Enthalpies
        if input_flow > 0:
            h_in = self.Inlet.getEnthalpy(temp=input_temp, mole_frac=input_fracs,
                                          basis='mole')
        else:
            h_in = 0

        h_liq = self.LiqEvap.getEnthalpy(temp, mole_frac=x_i, basis='mole')
        h_vap = self.VapEvap.getEnthalpy(temp, mole_frac=y_i, basis='mole')

        # Heat transfer
        diam = 0.438
        height_liq = vol_liq / (np.pi/4 * diam**2)
        area_ht = np.pi * diam * height_liq  # m**2

        heat_transfer = -self.ht_coeff * area_ht * (temp - self.temp_ht)

        # Compute balances
        diff_bce = heat_transfer + input_flow * h_in - flow_vap * h_vap - du_dt
        pv_term = pres * self.vol_tot
        internal_energy = mol_liq * h_liq + mol_vap * h_vap - pv_term - u_int

        out_energy = np.array([diff_bce, internal_energy])

        return out_energy

    def unit_model(self, time, states, states_dot, params=None):

        n_comp = self.num_species + 1

        # Decompose states
        moles_i = states[:n_comp]

        fracs = states[n_comp:3*n_comp]
        x_liq = fracs[:n_comp]
        y_vap = fracs[n_comp:]

        mol_liq = states[3*n_comp]
        mol_vap = states[3*n_comp + 1]

        pres = states[3*n_comp + 2]

        u_int = states[-2]
        temp = states[-1]

        # Decompose derivatives
        dmoli_dt = states_dot[:n_comp]
        du_dt = states_dot[-2]

        # Inputs
        u_inputs = self.get_inputs(time)

        # Material balance
        material_bces = self.material_balances(dmoli_dt,
                                               moles_i,
                                               x_liq, y_vap, mol_liq, mol_vap,
                                               pres, temp,
                                               u_inputs)

        material_bce = np.concatenate(material_bces[:-2])
        flow_vap, vol_liq = material_bces[-2:]

        # Energy balance
        energy_bce = self.energy_balances(du_dt, u_int, temp, x_liq, y_vap,
                                          mol_liq, mol_vap,
                                          vol_liq, flow_vap,
                                          pres, u_inputs)

        # Concatenate balances
        balances = np.concatenate((material_bce, energy_bce))

        # Update output objects
        self.LiqEvap.temp = temp
        self.VapEvap.temp = temp

        self.LiqEvap.mole_frac = x_liq
        self.VapEvap.mole_flow = y_vap

        return balances

    def unit_jacobian(self, c, time, states, sdot, params=None):
        jac_states = self.jac_states(time, states, sdot, params)
        jac_sdot = self.jac_sdot(time, states, sdot, params)

        jac_system = jac_states + c * jac_sdot

        return jac_system

    def init_unit(self):
        temp_init = self.LiqPhase.temp
        pres_init = self.pres

        # Mole fractions
        x_seed = self.LiqPhase.mole_frac
        x_seed = np.append(x_seed, 0)

        y_seed = np.zeros(self.num_species + 1)
        y_seed[-1] = 1

        # Moles of phases
        mol_liq = self.LiqPhase.moles
        vol_liq = self.LiqPhase.vol

        vol_vap = self.vol_tot - vol_liq
        if vol_vap < 0:
            raise ValueError(r"Drum volume ({:.2f} m3) lower than the liquid "
                             r"volume ({:.2f} m3)".format(self.vol_tot, vol_liq))
        mol_vap = self.pres * vol_vap / gas_ct / temp_init
        mol_tot = mol_liq + mol_vap

        # Moles of i
        moli_seed = mol_liq * x_seed + mol_vap * y_seed

        # ---------- Create new phase with inert gas and run adiabatic flash
        z_flash = moli_seed / moli_seed.sum()
        pres_in = self.LiqPhase.pres

        self.LiqEvap = LiquidPhase(self.paths, temp_init, pres=pres_in,
                                   moles=mol_tot, mole_frac=z_flash)

        FlashInit = AdiabaticFlash(pres_drum=pres_init,
                                   gamma_method=self.activity_model)
        FlashInit.Inlet = self.LiqEvap
        FlashInit.solve_unit()

        # Update phases and initial states with flash results
        self.LiqEvap = FlashInit.LiquidOut
        self.VapEvap = FlashInit.VaporOut

        x_init = FlashInit.LiquidOut.mole_frac
        y_init = FlashInit.VaporOut.mole_frac
        temp_init = FlashInit.LiquidOut.temp

        mol_liq = FlashInit.LiquidOut.moles
        dens_liq = self.LiqEvap.getDensity(basis='mole')
        vol_liq = mol_liq / dens_liq / 1000
        vol_vap = self.vol_tot - vol_liq

        mol_vap = pres_init * vol_vap / gas_ct / temp_init

        mol_vent = self.VapEvap.moles - mol_vap

        mol_i = mol_liq * x_init + mol_vap * y_init
        mol_tot = mol_liq + mol_vap

        if self.Inlet is None:
            dm_init = np.zeros_like(x_init)
            inlet_flow = 0
            hin_init = 0
        else:
            dm_init = self.Inlet.mole_flow * self.Inlet.mole_frac
            inlet_flow = self.Inlet.mole_flow
            hin_init = self.Inlet.getEnthalpy(basis='mole')

        # ---------- Energy balance states
        # Enthalpies

        hliq_init = self.LiqEvap.getEnthalpy(temp_init, basis='mole')
        hvap_init = self.VapEvap.getEnthalpy(temp_init, basis='mole')

        diam = 0.438
        height_liq = vol_liq / (np.pi/4 * diam**2)
        area_ht = np.pi * diam * height_liq  # m**2

        ht_init = -self.ht_coeff * area_ht * (temp_init - self.temp_ht)

        du_init = ht_init + inlet_flow * hin_init  # bce - dU_dt

        # Internal energy
        u_init = mol_liq * hliq_init + mol_vap * hvap_init - \
            pres_init * self.vol_tot

        # ---------- Retrieve results
        states_init = np.concatenate(
            (mol_i, x_init, y_init, [mol_liq, mol_vap],
             [pres_init, u_init, temp_init])
            )

        sdot_init = np.zeros_like(states_init)
        sdot_init[:self.num_species + 1] = dm_init
        sdot_init[-2] = du_init

        return states_init, sdot_init

    def solve_unit(self, runtime, solve=True):
        states_initial, sdev_initial = self.init_unit()

        # ---------- Solve
        n_comp = self.num_species + 1
        alg_map = np.concatenate((np.ones(n_comp),
                                  np.zeros(2*n_comp + 3),
                                  [1, 0])
                                 )

        # ---------- Solve problem
        if solve:
            # Create problem
            problem = Implicit_Problem(self.unit_model,
                                       states_initial, sdev_initial,
                                       t0=0)

            problem.algvar = alg_map
            # problem.jac = self.unit_jacobian

            # Set solver
            solver = IDA(problem)
            solver.make_consistent('IDA_YA_YDP_INIT')
            solver.suppress_alg = True

            # Solve
            time, states, sdot = solver.simulate(runtime)

            self.retrieve_results(time, states)

            return time, states
        else:
            fun_eval = self.unit_model(0, states_initial, sdev_initial)
            return fun_eval

    def retrieve_results(self, time, states):
        n_comp = self.num_species + 1

        self.timeProf.append(time)

        fracs = states[:, n_comp:3*n_comp]
        self.xliqProf.append(fracs[:, :n_comp])
        self.yvapProf.append(fracs[:, n_comp:])

        self.molLiqProf = states[:, 3*n_comp]
        self.molVapProf = states[:, 3*n_comp + 1]

        self.presProf.append(states[:, 3*n_comp + 2])

        self.uIntProf = states[:, -2]
        self.tempProf.append(states[:, -1])

        # Update phases
        self.LiqPhase.temp = self.tempProf[-1][-1]
        self.LiqPhase.pres = self.presProf[-1][-1]
        self.LiqPhase.updatePhase(mole_frac=self.xliqProf[-1][-1, :-1],
                                  moles=self.molLiqProf[-1])

        self.Phases = self.LiqPhase

        # Output info
        self.Outlet = self.LiqPhase
        self.outputs = np.column_stack((self.xliqProf[-1][:, :-1],
                                        self.molLiqProf, self.tempProf[-1]))

    def flatten_states(self):
        if type(self.timeProf) is list:
            self.xliqProf = np.vstack(self.xliqProf)
            self.yvapProf = np.vstack(self.yvapProf)

            # self.volProf = np.concatenate(self.volProf)
            self.tempProf = np.concatenate(self.tempProf)
            self.presProf = np.concatenate(self.presProf)
            self.timeProf = np.concatenate(self.timeProf)

            # if 'temp_ht' in self.states_uo:
            #     self.tempHtProf = np.concatenate(self.tempHtProf)

            # self.Phases.tempProf = self.tempProf
            # self.Phases.concProf = self.concProf
            # self.Phases.timeProf = self.timeProf

    def plot_profiles(self, fig_size=None, pick_comp=None, time_div=1):
        self.flatten_states()

        if pick_comp is None:
            pick_comp = np.arange(self.num_species)
        else:
            pick_comp = pick_comp

        # Fractions
        time_plot = self.timeProf / time_div
        fig, ax = plt.subplots(2, 2, figsize=fig_size)
        ax[0, 0].plot(time_plot, self.xliqProf[:, pick_comp])
        ax[0, 1].plot(time_plot, self.yvapProf[:, pick_comp])

        ax[0, 0].set_ylabel('$x_i$')
        ax[0, 1].set_ylabel('$y_i$')

        leg = [self.LiqEvap.name_species[ind] for ind in pick_comp]
        ax[0, 0].legend(leg)

        # T and P
        ax[1, 0].plot(time_plot, self.tempProf, 'k')

        ax_pres = ax[1, 0].twinx()

        color = 'r'
        ax_pres.plot(time_plot, self.presProf/1000, color)

        ax_pres.spines['right'].set_color(color)
        ax_pres.tick_params(colors=color)
        ax_pres.yaxis.label.set_color(color)

        ax[1, 0].set_ylabel('$T$ (K)')
        ax_pres.set_ylabel('$P$ (kPa)')

        # Moles
        ax[1, 1].plot(time_plot, self.molLiqProf, 'k')
        ax_vap = ax[1, 1].twinx()
        ax_vap.plot(time_plot, self.molVapProf, color)

        ax_vap.spines['right'].set_color(color)
        ax_vap.tick_params(colors=color)
        ax_vap.yaxis.label.set_color(color)

        ax[1, 1].set_ylabel('$M_L$ (mol)')
        ax_vap.set_ylabel('$M_V$ (mol)')

        for axis in ax.flatten():
            axis.grid(which='both')

            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)

            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        for axis in [ax_vap, ax_pres]:
            axis.spines['top'].set_visible(False)
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        if time_div == 1:
            fig.text(0.5, 0, 'time (s)', ha='center')

        fig.tight_layout()

        return fig, ax


class ContinuousEvaporator:
    def __init__(self, vol_drum, phase=None, inlet=None, adiabatic=True,
                 pres=101325, diam_out=2.54e-2, frac_liq=0.5, k_liq=1,
                 ht_coeff=1000, temp_ht=298.15,
                 activity_model='ideal'):

        self._Inlet = inlet
        self._Phases = phase

        self.vol_tot = vol_drum
        self.vol_liq_set = frac_liq * vol_drum
        self.k_liq = k_liq
        self.adiabatic = adiabatic

        self.diam_tank = (4/np.pi * self.vol_liq_set)**(1/3)

        self.area_out = np.pi / 4 * diam_out**2

        self.pres = pres

        # Jacobians
        self.jac_states = jacauto(self.unit_model, 1)
        self.jac_sdot = jacauto(self.unit_model, 2)

        # Heat transfer
        self.ht_coeff = ht_coeff
        self.temp_ht = temp_ht

        self.oper_mode = 'Continuous'

        self.is_continuous = True

        self.timeProf = []
        self.tempProf = []
        self.presProf = []

        self.xliqProf = []
        self.yvapProf = []

        self.activity_model = activity_model

        self.nomenclature()

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phase):
        path_comp = phase.path_data

        self.LiqPhase = phase

        self.num_species = len(self.LiqPhase.mole_frac)
        self.name_species = self.LiqPhase.name_species

        self.VapPhase = VaporStream(path_comp, pres=self.pres)

        self._Phases = phase

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):  # Create an inlet with additional species (N2)

        self._Inlet = inlet

        self.oper_mode = 'Semibatch'

    def nomenclature(self):
        self.names_states_in = ['mole_frac', 'mole_flow', 'temp']
        self.names_states_out = ['mole_frac', 'mole_flow', 'temp']

    def get_inputs(self, time):
        if self.Inlet.y_upstream is None or len(self.Inlet.y_upstream) == 1:
            input_dict = {'mole_flow': self.Inlet.mole_flow,
                          'temp': self.Inlet.temp,
                          'mole_frac': self.Inlet.mole_frac}
        else:
            all_inputs = self.Inlet.InterpolateInputs(time)

            inputs = get_dict_states(self.names_upstream, self.num_species,
                                     0, all_inputs)

            input_dict = {}
            for name in self.names_states_in:
                input_dict[name] = inputs[self.bipartite[name]]

        return input_dict

    def material_balances(self, moles_i, x_i, y_i,
                          mol_liq, mol_vap, pres, temp, u_inputs,
                          flows_out=False):

        input_flow = u_inputs['mole_flow']
        input_fracs = u_inputs['mole_frac']

        mw_vap = np.dot(self.VapPhase.mw, y_i.T)

        rho_liq = self.LiqPhase.getDensity(mole_frac=x_i, temp=temp,
                                           basis='mole')

        vol_liq = mol_liq / rho_liq / 1000  # m**3
        vol_vap = mol_vap * gas_ct * temp / pres
        vol_eqn = vol_liq + vol_vap - self.vol_tot

        rho_gas = pres / gas_ct / temp * (mw_vap / 1000)  # kg/m**3
        flow_vap = self.area_out / mw_vap * 1000 *\
            np.sqrt(2*rho_gas * np.maximum(eps, pres - self.pres))  # mol/s

        # flow_vap = 0.1 * (pres - self.pres)

        flow_liq = self.k_liq * (vol_liq - self.vol_liq_set) * rho_liq * 1000 \
            + input_flow*1  # mol/s

        if flows_out:
            return flow_liq, flow_vap, vol_liq
        else:
            # Differential eqns
            dmoli_dt = input_flow * input_fracs - flow_liq * x_i - flow_vap * y_i

            # Algebraic eqns
            component_bce = mol_liq * x_i + mol_vap * y_i - moles_i
            global_bce = mol_liq + mol_vap - sum(moles_i)

            k_i = self.LiqPhase.getKeqVLE(temp, pres, x_i, self.activity_model)
            equilibria = y_i - k_i * x_i

            p_sat = self.LiqPhase.AntoineEquation(temp=temp)
            pres_eqn = np.dot(x_i, p_sat) - pres

            alg_balances = np.concatenate((component_bce,  # x_i
                                           equilibria,  # y_i
                                           np.array([global_bce]),  # M_L
                                           np.array([vol_eqn]),  # M_V
                                           np.array([pres_eqn])  # P
                                           )
                                          )

            return dmoli_dt, alg_balances, flow_liq, flow_vap, vol_liq

    def energy_balances(self, u_int, temp, x_i, y_i, mol_liq, mol_vap, vol_liq,
                        flow_liq, flow_vap, pres, u_inputs):

        input_flow = u_inputs['mole_flow']
        input_fracs = u_inputs['mole_frac']
        input_temp = u_inputs['temp']

        # Enthalpies
        h_in = self.Inlet.getEnthalpy(temp=input_temp, mole_frac=input_fracs,
                                      basis='mole')

        h_liq = self.LiqPhase.getEnthalpy(temp, mole_frac=x_i, basis='mole')
        h_vap = self.VapPhase.getEnthalpy(temp, mole_frac=y_i, basis='mole')

        # Heat transfer
        if self.adiabatic:
            heat_transfer = 0
        else:
            height_liq = vol_liq / (np.pi/4 * self.diam_tank**2)
            area_ht = np.pi * self.diam_tank * height_liq  # m**2

            heat_transfer = self.ht_coeff * area_ht * (temp - self.temp_ht)

        # Compute balances
        duint_dt = input_flow * h_in - flow_liq * h_liq - flow_vap * h_vap - \
            heat_transfer

        pv_term = pres * self.vol_tot
        internal_energy = mol_liq * h_liq + mol_vap * h_vap - pv_term - u_int

        out_energy = np.array([duint_dt, internal_energy])

        return out_energy

    def unit_model(self, time, states, states_dot=None, params=None):

        n_comp = self.num_species

        # Decompose states
        moles_i = states[:n_comp]

        fracs = states[n_comp:3*n_comp]
        x_liq = fracs[:n_comp]
        y_vap = fracs[n_comp:]

        mol_liq = states[3*n_comp]
        mol_vap = states[3*n_comp + 1]

        pres = states[3*n_comp + 2]

        u_int = states[-2]
        temp = states[-1]

        # Inputs
        u_inputs = self.get_inputs(time)

        # Material balance
        material_bces = self.material_balances(moles_i,
                                               x_liq, y_vap, mol_liq, mol_vap,
                                               pres, temp,
                                               u_inputs)

        material_bce = np.concatenate(material_bces[:-3])
        flow_liq, flow_vap, vol_liq = material_bces[-3:]

        # Energy balance
        energy_bce = self.energy_balances(u_int, temp, x_liq, y_vap,
                                          mol_liq, mol_vap, vol_liq,
                                          flow_liq, flow_vap,
                                          pres, u_inputs)

        # Concatenate balances
        balances = np.concatenate((material_bce, energy_bce))

        if states_dot is not None:
            # Decompose derivatives
            dmolesi_dt = states_dot[:n_comp]
            duint_dt = states_dot[-2]

            balances[:n_comp] = balances[:n_comp] - dmolesi_dt
            balances[-2] = balances[-2] - duint_dt

        # Update output objects
        self.LiqPhase.temp = temp
        self.VapPhase.temp = temp

        self.LiqPhase.mole_frac = x_liq
        self.VapPhase.mole_flow = y_vap

        return balances

    def unit_jacobian(self, c, time, states, sdot, params=None):
        jac_states = self.jac_states(time, states, sdot, params)
        jac_sdot = self.jac_sdot(time, states, sdot, params)

        jac_system = jac_states + c * jac_sdot

        return jac_system

    def init_unit(self):
        temp_bubble_init, y_init = self.LiqPhase.getBubblePoint(pres=self.pres,
                                                                y_vap=True)
        pres_init = self.pres

        # Mole fractions
        x_init = self.LiqPhase.mole_frac
        # x_seed = np.append(x_seed, 0)

        # y_ = np.zeros(self.num_species + 1)
        # y_seed[-1] = 1

        # Moles of phases
        mol_liq = self.LiqPhase.moles
        vol_liq = self.LiqPhase.vol

        vol_vap = self.vol_tot - vol_liq
        if vol_vap < 0:
            raise ValueError(r"Drum volume ({:.2f} m3) lower than the liquid "
                             r"volume ({:.2f} m3)".format(self.vol_tot,
                                                          vol_liq))
        mol_vap = self.pres * vol_vap / gas_ct / temp_bubble_init

        self.VapPhase.updatePhase(moles=mol_vap, mole_frac=y_init)

        # Moles of i
        mol_i = mol_liq * x_init + mol_vap * y_init

        u_inlet = self.get_inputs(0)
        inlet_flow = u_inlet['mole_flow']

        # Liquid flow
        rho_liq = self.LiqPhase.getDensity(mole_frac=x_init,
                                           temp=temp_bubble_init,
                                           basis='mole')

        flow_liq = self.k_liq * (vol_liq - self.vol_liq_set) * rho_liq * 1000 \
            + inlet_flow*1  # mol/s

        dm_init = inlet_flow * u_inlet['mole_frac'] - flow_liq * x_init

        # ---------- Energy balance states
        # Enthalpies
        hin_init = self.Inlet.getEnthalpy(basis='mole', temp=u_inlet['temp'])
        hliq_init = self.LiqPhase.getEnthalpy(temp_bubble_init, basis='mole')
        hvap_init = self.VapPhase.getEnthalpy(temp_bubble_init, basis='mole',
                                              mole_frac=y_init)

        # Disregard vapor flow at the beginning (vapor phase is at P_0)
        du_init = inlet_flow * hin_init - flow_liq * hliq_init   # bce - dU_dt

        # Internal energy
        u_init = mol_liq * hliq_init + mol_vap * hvap_init - \
            pres_init * self.vol_tot

        # ---------- Retrieve results
        states_init = np.concatenate(
            (mol_i, x_init, y_init, [mol_liq, mol_vap],
             [pres_init, u_init, temp_bubble_init])
            )

        sdot_init = np.zeros_like(states_init)
        sdot_init[:self.num_species] = dm_init
        sdot_init[-2] = du_init

        return states_init, sdot_init

    def solve_unit(self, runtime, solve=True, steady_state=False):
        states_initial, sdev_initial = self.init_unit()

        # ---------- Solve
        n_comp = self.num_species
        alg_map = np.concatenate((np.ones(n_comp),
                                  np.zeros(2*n_comp + 3),
                                  [1, 0])
                                 )

        # ---------- Solve problem
        if steady_state:
            obj_fn = lambda states: self.unit_model(0, states)
            steady_solution = fsolve(obj_fn, states_initial)

            return steady_solution
        elif solve:
            # Create problem
            problem = Implicit_Problem(self.unit_model,
                                       states_initial, sdev_initial,
                                       t0=0)

            problem.algvar = alg_map
            # problem.jac = self.unit_jacobian

            # Set solver
            solver = IDA(problem)
            solver.make_consistent('IDA_YA_YDP_INIT')
            solver.suppress_alg = True

            # Solve
            time, states, sdot = solver.simulate(runtime)

            self.retrieve_results(time, states)

            return time, states
        else:
            fun_eval = self.unit_model(0, states_initial, sdev_initial)
            return fun_eval

    def retrieve_results(self, time, states):
        n_comp = self.num_species

        self.timeProf.append(time)

        fracs = states[:, n_comp:3*n_comp]
        self.xliqProf.append(fracs[:, :n_comp])
        self.yvapProf.append(fracs[:, n_comp:])

        self.molLiqProf = states[:, 3*n_comp]
        self.molVapProf = states[:, 3*n_comp + 1]

        self.presProf.append(states[:, 3*n_comp + 2])

        self.uIntProf = states[:, -2]
        self.tempProf.append(states[:, -1])

        # Update phases
        self.LiqPhase.temp = self.tempProf[-1][-1]
        self.LiqPhase.pres = self.presProf[-1][-1]
        self.LiqPhase.updatePhase(mole_frac=self.xliqProf[-1][-1],
                                  moles=self.molLiqProf[-1])

        self.Phases = self.LiqPhase

        inputs_all = self.get_inputs(time)
        flow_liq, flow_vap, vol_liq = self.material_balances(
            states[:, :n_comp],
            self.xliqProf[-1], self.yvapProf[-1],
            self.molLiqProf, self.molVapProf,
            self.presProf[-1], self.tempProf[-1],
            inputs_all, flows_out=True)

        self.flowLiqProf = flow_liq
        self.flowVapProf = flow_vap
        self.volLiqProf = vol_liq

        # Output info
        self.Outlet = LiquidStream(self.LiqPhase.path_data,
                                   temp=self.tempProf[-1][-1],
                                   pres=self.presProf[-1][-1],
                                   mole_frac=self.xliqProf[-1][-1],
                                   mole_flow=flow_liq[-1])

        self.outputs = np.column_stack((self.xliqProf[-1],
                                        self.flowLiqProf, self.tempProf[-1]))

    def flatten_states(self):
        if type(self.timeProf) is list:
            self.xliqProf = np.vstack(self.xliqProf)
            self.yvapProf = np.vstack(self.yvapProf)

            # self.volProf = np.concatenate(self.volProf)
            self.tempProf = np.concatenate(self.tempProf)
            self.presProf = np.concatenate(self.presProf)
            self.timeProf = np.concatenate(self.timeProf)

            # if 'temp_ht' in self.states_uo:
            #     self.tempHtProf = np.concatenate(self.tempHtProf)

            # self.Phases.tempProf = self.tempProf
            # self.Phases.concProf = self.concProf
            # self.Phases.timeProf = self.timeProf

    def plot_profiles(self, fig_size=None, pick_comp=None, time_div=1,
                      vol_plot=True):
        self.flatten_states()

        if pick_comp is None:
            pick_comp = np.arange(self.num_species)
        else:
            pick_comp = pick_comp

        # Fractions
        time_plot = self.timeProf / time_div
        fig, ax = plt.subplots(2, 2, figsize=fig_size)
        ax[0, 0].plot(time_plot, self.xliqProf[:, pick_comp])
        ax[0, 1].plot(time_plot, self.yvapProf[:, pick_comp])

        ax[0, 0].set_ylabel('$x_i$')
        ax[0, 1].set_ylabel('$y_i$')

        leg = [self.name_species[ind] for ind in pick_comp]
        ax[0, 0].legend(leg)

        # T and P
        ax[1, 0].plot(time_plot, self.tempProf, 'k')

        ax_pres = ax[1, 0].twinx()

        color = 'r'
        ax_pres.plot(time_plot, self.presProf/1000, color)

        ax_pres.spines['right'].set_color(color)
        ax_pres.tick_params(colors=color)
        ax_pres.yaxis.label.set_color(color)

        ax[1, 0].set_ylabel('$T$ (K)')
        ax_pres.set_ylabel('$P$ (kPa)')

        # Moles or volume
        if vol_plot:
            ax[1, 1].plot(time_plot, self.volLiqProf, 'k')
            ax_vap = ax[1, 1].twinx()
            ax_vap.plot(time_plot, self.vol_tot - self.volLiqProf, color)

            ax[1, 1].set_ylabel('$V_L$ ($m^3$)')
            ax_vap.set_ylabel('$V_V$ ($m^3$)')
        else:
            ax[1, 1].plot(time_plot, self.molLiqProf, 'k')
            ax_vap = ax[1, 1].twinx()
            ax_vap.plot(time_plot, self.molVapProf, color)

            ax[1, 1].set_ylabel('$M_L$ (mol)')
            ax_vap.set_ylabel('$M_V$ (mol)')

        ax_vap.spines['right'].set_color(color)
        ax_vap.tick_params(colors=color)
        ax_vap.yaxis.label.set_color(color)

        for axis in ax.flatten():
            axis.grid(which='both')

            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)

            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        for axis in [ax_vap, ax_pres]:
            axis.spines['top'].set_visible(False)
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        if time_div == 1:
            fig.text(0.5, 0, 'time (s)', ha='center')

        fig.tight_layout()

        return fig, ax

