# -*- coding: utf-8 -*-


import numpy as np
# from autograd import numpy as np
# from autograd import jacobian as jacauto
from PharmaPy.Commons import (mid_fn, trapezoidal_rule, eval_state_events,
                              handle_events, unpack_states, flatten_states)
from PharmaPy.Connections import get_inputs, get_inputs_new
from PharmaPy.Streams import LiquidStream, VaporStream
from PharmaPy.Phases import LiquidPhase, VaporPhase, classify_phases

from PharmaPy.Results import DynamicResult
from PharmaPy.Plotting import plot_function

from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
from assimulo.exception import TerminateSimulation

from pathlib import Path
import copy

gas_ct = 8.314
eps = np.finfo(float).eps


def merge_supercritical(flags, x_liq, y_vap, z_super):
    x = np.zeros(len(x_liq) + sum(flags))
    y = np.zeros_like(x)

    x[~flags] = x_liq

    y[flags] = (1 - y_vap.sum()) * z_super / z_super.sum()
    y[~flags] = y_vap

    return x, y


class IsothermalFlash:
        
    """
    Create an isothermal flash object. The model run by the instances of
    this class is steady state isothermal flash given T and P. This
    model is often used to determine the thermodynamic state of a given
    stream or phase (only liquid, vapor-liquid, only vapor)

    Parameters
    ----------
    temp_drum : float, optional
        equipment temperature [K]. If None, temperature will be taken from
        the Inlet object aggregated to the flash (from PharmaPy.Phases or
        PharmaPy.Streams modules). The default is None.
    pres_drum : float, optional
        equipment pressure [Pa]. If None, pressure will be taken from
        the Inlet object aggregated to the flash (from PharmaPy.Phases or
        PharmaPy.Streams modules). The default is None.
    gamma_method : str, optional
        one of 'ideal', 'UNIFAC' or 'UNIQUAC'. If 'UNIFAC' or 'UNIQUAC' is
        passed, the pure-component .json file must have the interaction
        parameters of the model. The default is 'ideal'.

    Returns
    -------
    None.

    """

    def __init__(self, temp_drum=None, pres_drum=None,gamma_method='ideal'):

        self.temp = temp_drum
        self.pres = pres_drum
        self._Inlet = None
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

        self.result = {'phi': solution[[0, -1]],
                       'x_out': x_flash, 'y_out': y_flash}

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
    """
    Create an adiabatic flash object.

    Parameters
    ----------
    pres_drum : float
        tank pressure [Pa].
    div_energybce : float, optional
        Constant by which the energy balance equation is divided.
        It helps improving the conditioning of the numerical solver.
        The default is 1e3.
    gamma_method : str, optional
        one of 'ideal', 'UNIFAC' or 'UNIQUAC'. If 'UNIFAC' or 'UNIQUAC' is
        passed, the pure-component .json property file must have required
        parameters for the activity coefficient model. 
    mult_midfun : TYPE, optional  
        DESCRIPTION. The default is 1.
    seed_basedon_input : TYPE, optional  
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    def __init__(self, pres_drum, div_energybce=1e3, gamma_method='ideal',
                 mult_midfun=1, seed_basedon_input=False):
 
        self.pres = pres_drum

        self.div_energybce = div_energybce
        self.mult_midfun = mult_midfun
        self.seed_basedon_input = seed_basedon_input

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
        vap_flow = mid_fn(args_mid) * self.mult_midfun

        if any(self.is_supercritic):
            comp_bces = comp_bces[~self.is_supercritic]
            equilibria = equilibria[~self.is_supercritic]

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
        energy_bce = np.atleast_1d(energy_bce / self.div_energybce)

        return energy_bce

    def unit_model(self, states):
        liq = states[0]
        vap = states[-2]

        fracs = states[1:-2]
        x_i = fracs[:self.num_comp]
        y_i = fracs[self.num_comp:]
        z_i = self.Inlet.mole_frac

        if any(self.is_supercritic):
            x_i, y_i = merge_supercritical(self.is_supercritic, x_i, y_i,
                                           self.z_super)

        temp = states[-1]

        material_bces = self.material_balances(liq, vap, x_i, y_i, z_i, temp)
        energy_bce = self.energy_balance(liq, vap, x_i, y_i, z_i, temp)

        balances = np.concatenate((material_bces, energy_bce))

        return balances

    def solve_unit(self, v_seed=0.5):
        """ Solve AdiabaticFlash unit


        Parameters
        ----------
        v_seed : float, optional
            A seed for output fraction of vapor with respect to feed material
            to the flash. It must be in the range 0-1. The default is 0.5.

        Returns
        -------
        solution : SciPy OptimizerResult object
            solution of the root finding algorithm.

        """

        if self.seed_basedon_input:
            x_seed = self.Inlet.mole_frac
            y_seed = x_seed * self.Inlet.getKeqVLE(pres=self.pres)
            y_seed = y_seed / y_seed.sum()
        else:
            x_seed = np.ones(self.num_comp) * 1 / self.num_comp
            y_seed = x_seed

        is_supercritic = self.Inlet.temp > self.Inlet.t_crit

        if any(is_supercritic):
            x_seed = x_seed[~is_supercritic]
            y_seed = y_seed[~is_supercritic]

            self.num_comp -= sum(is_supercritic)
            self.z_super = self.Inlet.mole_frac[is_supercritic]

        self.is_supercritic = is_supercritic

        l_seed = 1 - v_seed

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

        if any(is_supercritic):
            x_flash, y_flash = merge_supercritical(self.is_supercritic,
                                                   x_flash, y_flash,
                                                   self.z_super)

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
    """ 
    Create a batch/semibatch evaporator object
    
    Parameters
    ----------
    vol_drum : float
        total drum volume [m**3].
    pressure : float, optional
        pressure setpoint [Pa] (actual pressure is computed by the
        evaporator model). The default is 101325.
    diam_out : float, optional
        diameter of the vapor outlet pipe [m]. The default is 2.54e-2.
    k_vap : float, optional
        constant for vapor flow proportional control. Vapor flow is
        calculated as F_v = k_vap * (P_model - P) + F_in. The default is 1.
    h_conv : float, optional
        convective heat transfer coefficient for the liquid phase
        [W/m**2/K]. The default is 1000.
    activity_model : str, optional
        model to be used for calculation activity coefficient in
        vapor-liquid equilibria. Choose one of 'ideal', 'UNIFAC' and
        'UNIQUAC' The default is 'ideal'.
    state_events : list of dicts, optional
        list of dictionaries containing the specification of a state event.
        To learn about the structure of a state event, see the
        PharmaPy.Commons.eval_state_events documentation.
        The default is None.
    stop_at_maxvol : bool, optional
        whether or not to automatically stop integration when liquid volume
        reaches tank volume. This can be important for semi-batch
        vaporization. is important for
        The default is True.
    flash_kwargs : dict, optional
        dictionary to be passed to the solve_unit method of the
        PharmaPy.AdiabaticFlash instance run to initialize the vaporizer.
        The default is None.

    Returns
    -------
    A vaporizer object (VO). If a PharmaPy.Stream object is aggregated to the
    resulting instance (instance.Inlet = PharmaPy.Stream(...)),
    the VO will be interpreted as a Semi-batch evaporator object.
    Otherwise, a Batch evaporator will be run.

    """
    def __init__(self, vol_drum,
                 pressure=101325, diam_out=2.54e-2,
                 k_vap=1, cv_gas=0.8,
                 h_conv=1000,
                 activity_model='ideal', state_events=None,
                 stop_at_maxvol=True, flash_kwargs=None,
                 include_nitrogen=False):


        self._Inlet = None
        self._Phases = None
        self.include_nitrogen = include_nitrogen

        self._Utility = None
        self.material_from_upstream = False

        self.k_vap = k_vap
        self.cv_gas = cv_gas

        if isinstance(state_events, list):
            self.state_events = state_events
        elif isinstance(state_events, dict):
            self.state_events = [state_events]
        elif state_events is None:
            self.state_events = []

        self.stop_at_maxvol = stop_at_maxvol

        if flash_kwargs is None:
            self.flash_kwargs = {}
        else:
            self.flash_kwargs = flash_kwargs

        # Geometry
        self.vol_tot = vol_drum
        self.diam_tank = (4/np.pi * vol_drum)**(1/3)
        self.area_base = np.pi / 4 * self.diam_tank**2
        self.area_out = np.pi / 4 * diam_out**2

        self.pres = pressure

        # # Jacobians
        # self.jac_states = jacauto(self.unit_model, 1)
        # self.jac_sdot = jacauto(self.unit_model, 2)

        # Heat transfer
        self.h_conv = h_conv

        self.oper_mode = 'Batch'  # If inlet setter, then Semibatch

        self.is_continuous = False

        self.profiles_runs = []

        self.activity_model = activity_model

        self.nomenclature()

        self.vol_offset = 1

        self.elapsed_time = 0
        self.allow_flow = True

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phase):
        path_comp = phase.path_data
        path_inert = '/data/evaporator/props_nitrogen.json'
        path_inert = str(Path(__file__).parents[1]) + path_inert
        paths = [path_comp, path_inert]
        self.paths = paths

        self.__original_phase__ = copy.deepcopy(phase)

        liquid = phase
        vapor = VaporStream(path_comp, pres=self.pres,
                            mole_frac=liquid.mole_frac,
                            mole_flow=eps)

        self._Phases = [liquid, vapor]
        classify_phases(self)

        self.num_species = len(self.Liquid_1.mole_frac)
        self.name_species = phase.name_species

        self.nomenclature()

        self.states_di = {
            'mol_i': {'index': self.name_species, 'units': 'mol',
                      'dim': len(self.name_species), 'type': 'diff'},
            'x_liq': {'index': self.name_species, 'units': '',
                      'dim': len(self.name_species), 'type': 'alg'},
            'y_vap': {'index': self.name_species, 'units': '',
                      'dim': len(self.name_species), 'type': 'alg'},
            'mol_liq': {'units': 'mol', 'dim': 1, 'type': 'alg'},
            'mol_vap': {'units': 'mol', 'dim': 1, 'type': 'alg'},
            'pres': {'units': 'Pa', 'dim': 1, 'type': 'alg'},
            'u_int': {'units': 'J', 'dim': 1, 'type': 'diff'},
            'temp': {'units': 'K', 'dim': 1, 'type': 'alg'},
            }

        self.fstates_di = {
            'vol_liq': {'units': 'm^3', 'dim': 1},
            'vol_vap': {'units': 'm^3', 'dim': 1}
                }

        self.name_states = list(self.states_di.keys())

        self.dim_states = [di['dim'] for di in self.states_di.values()]

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):  # Create an inlet with additional species (N2)

        fields = ['temp', 'pres', 'mole_flow', 'mole_frac',
                  'controls', 'args_control']
        inlet_dict = {key: inlet.__dict__.get(key) for key in fields}
        inlet_dict['mole_frac'] = np.append(inlet_dict['mole_frac'], 0)

        self.inlet_inert_dict = inlet_dict

        self._Inlet = inlet

        self.oper_mode = 'Semibatch'

    @property
    def Utility(self):
        return self._Utility

    @Utility.setter
    def Utility(self, utility):
        self.u_ht = 1 / (1 / self.h_conv + 1 / utility.h_conv)
        self._Utility = utility

    def nomenclature(self):
        self.names_states_in = ['mole_frac', 'mole_flow', 'temp']
        self.names_states_out = ['mole_frac', 'moles', 'temp']

        self.names_upstream = None
        self.bipartite = None

    def get_inputs(self, time):

        if self.Inlet is None:
            inputs = {'mole_flow': 0,
                      'mole_frac': np.zeros(
                          self.num_species + self.include_nitrogen),
                      'temp': 298.15}

            inputs['Inlet'] = inputs
        else:
            inputs = get_inputs_new(time, self.Inlet, self.states_in_dict)

        return inputs

    def material_balances(self, time, mol_i, x_liq, y_vap,
                          mol_liq, mol_vap, pres, u_int, temp, u_inputs,
                          dmoli_dt=None):

        input_flow = u_inputs['mole_flow']
        input_fracs = u_inputs['mole_frac']

        mw_vap = np.dot(self.Vapor_1.mw, y_vap)

        rho_mol = pres / gas_ct / temp  # mol/m**3
        rho_gas = rho_mol * (mw_vap / 1000)  # kg/m**3
        vel_vap = np.sqrt(np.maximum(eps, 2 * (pres - self.pres)/rho_gas))

        flow_vap = rho_mol * self.area_out * vel_vap * self.cv_gas * self.k_vap

        rho_liq = self.Liquid_1.getDensity(mole_frac=x_liq, temp=temp,
                                           basis='mole')

        vol_liq = mol_liq / rho_liq / 1000  # m**3
        vol_vap = mol_vap * gas_ct * temp / pres

        if dmoli_dt is None:
            return flow_vap, vol_liq
        else:
            # Differential eqns
            diff_i = input_flow * input_fracs - flow_vap * y_vap - dmoli_dt  # bce - dMi_dt

            # Algebraic eqns
            not_super = 1 - self.is_supercritic
            component_bce = mol_liq * x_liq * not_super + mol_vap * y_vap - mol_i
            global_bce = mol_liq + mol_vap - sum(mol_i)

            vol_eqn = vol_liq + vol_vap - self.vol_tot

            k_i = self.Liquid_1.getKeqVLE(temp, pres, x_liq, self.activity_model)

            equilibria = y_vap * not_super - k_i * x_liq
            equilibria = (y_vap - k_i * x_liq)[not_super.astype(bool)]
            # equilibria = y_vap - k_i * x_liq

            p_sat = self.Liquid_1.AntoineEquation(temp=temp) * not_super
            p_super = y_vap[self.is_supercritic] * pres

            pres_eqn = np.dot(p_sat, x_liq) + sum(p_super) - pres
            # pres_eqn = sum(y_vap - x_liq)

            alg_balances = np.concatenate((component_bce,  # x_liq
                                           equilibria,  # y_vap
                                           np.array([global_bce]),  # M_L
                                           np.array([vol_eqn]),  # M_V
                                           np.array([pres_eqn])  # P
                                           )
                                          )

            return diff_i, alg_balances, flow_vap, vol_liq

    def energy_balances(self, time, vol_liq, flow_vap, u_int, temp,
                        x_liq, y_vap, mol_liq, mol_vap, pres, mol_i,
                        u_inputs, du_dt=None):

        input_flow = u_inputs['mole_flow']
        input_fracs = u_inputs['mole_frac']
        input_temp = u_inputs['temp']

        # Enthalpies
        if isinstance(input_flow, np.ndarray):
            h_in = self.Inlet.getEnthalpy(temp=input_temp,
                                          mole_frac=input_fracs,
                                          basis='mole')
        else:
            if input_flow > 0:
                h_in = self.Inlet.getEnthalpy(temp=input_temp,
                                              mole_frac=input_fracs,
                                              basis='mole')
            else:
                h_in = 0

        h_liq = self.Liquid_1.getEnthalpy(temp, mole_frac=x_liq, basis='mole')
        h_vap = self.Vapor_1.getEnthalpy(temp, mole_frac=y_vap, basis='mole')

        # Heat transfer
        area_ht = 4 / self.diam_tank * vol_liq + self.area_base  # m**2
        temp_ht = self.Utility.get_inputs(time)['temp_in']

        heat_transfer = -self.u_ht * area_ht * (temp - temp_ht)

        if du_dt is None:
            return heat_transfer
        else:
            # Compute balances
            diff_bce = heat_transfer + input_flow * h_in - flow_vap * h_vap - du_dt
            pv_term = pres * self.vol_tot
            internal_energy = mol_liq * h_liq + mol_vap * h_vap - pv_term - u_int

            out_energy = np.array([diff_bce, internal_energy])

        return out_energy

    def unit_model(self, time, states, states_dot, sw, params=None):

        # Decompose states
        di_states = unpack_states(states, self.dim_states, self.name_states)

        # Decompose derivatives
        if states_dot is None:
            dmoli_dt = None
            du_dt = None
        else:
            di_dot = unpack_states(states_dot, self.dim_states,
                                   self.name_states)
            # dmoli_dt = states_dot[:n_comp]
            dmoli_dt = di_dot['mol_i']
            du_dt = di_dot['u_int']

        # Inputs
        u_inputs = self.get_inputs(time)['Inlet']

        u_inputs['mole_flow'] *= self.allow_flow

        if states_dot is None:
            material_bces = self.material_balances(time, **di_states,
                                                   u_inputs=u_inputs,
                                                   dmoli_dt=dmoli_dt)

            flow_vap, vol_liq = material_bces

            # Energy balance
            energy_bce = self.energy_balances(time, vol_liq, flow_vap,
                                              u_inputs=u_inputs,
                                              du_dt=du_dt,
                                              **di_states)

            return material_bces, energy_bce
        else:
            material_bces = self.material_balances(time, **di_states,
                                                   u_inputs=u_inputs,
                                                   dmoli_dt=dmoli_dt)

            material_bce = np.concatenate(material_bces[:-2])
            flow_vap, vol_liq = material_bces[-2:]

            # Energy balance
            energy_bce = self.energy_balances(time, vol_liq, flow_vap,
                                              u_inputs=u_inputs,
                                              du_dt=du_dt,
                                              **di_states)

            # Concatenate balances
            balances = np.concatenate((material_bce, energy_bce))

            # Update output objects (TODO: is this really necessary?)
            self.Liquid_1.temp = di_states['temp']
            self.Vapor_1.temp = di_states['temp']

            self.Liquid_1.mole_frac = di_states['x_liq']
            self.Vapor_1.mole_flow = di_states['y_vap']

            # print(abs(balances).max())

            return balances

    def unit_jacobian(self, c, time, states, sdot, params=None):
        jac_states = self.jac_states(time, states, sdot, params)
        jac_sdot = self.jac_sdot(time, states, sdot, params)

        jac_system = jac_states + c * jac_sdot

        return jac_system

    def init_unit(self):
        temp_init = self.Liquid_1.temp
        pres_init = self.pres

        x_init = self.Liquid_1.mole_frac
        y_init = np.zeros_like(x_init)

        # Moles of phases
        mol_liq = self.Liquid_1.moles
        vol_liq = self.Liquid_1.vol

        vol_vap = self.vol_tot - vol_liq
        if vol_vap < 0:
            raise ValueError(r"Drum volume ({:.2e} m3) lower than the liquid "
                             r"volume ({:.2e} m3)".format(self.vol_tot, vol_liq))

        mol_vap = self.pres * vol_vap / gas_ct / temp_init
        mol_tot = mol_liq + mol_vap

        if self.include_nitrogen:
            x_init = np.append(x_init, 0)

            y_init = np.append(y_init, 1)

            # Moles of i
            mol_i = mol_liq * x_init + mol_vap * y_init

            z_flash = mol_i / mol_i.sum()
            pres_in = self.Liquid_1.pres

            LiqEvap = LiquidPhase(self.paths, temp_init, pres=pres_in,
                                  moles=mol_tot, mole_frac=z_flash)

            FlashInit = AdiabaticFlash(pres_drum=pres_init,
                                       gamma_method=self.activity_model,
                                       **self.flash_kwargs)

            FlashInit.Inlet = LiqEvap
            FlashInit.solve_unit()

            # Update phases and initial states with flash results
            self.Liquid_1 = FlashInit.LiquidOut
            self.Vapor_1 = FlashInit.VaporOut

            x_init = FlashInit.LiquidOut.mole_frac
            y_init = FlashInit.VaporOut.mole_frac

            temp_init = FlashInit.LiquidOut.temp

            mol_liq = FlashInit.LiquidOut.moles
            dens_liq = self.Liquid_1.getDensity(basis='mole')
            vol_liq = mol_liq / dens_liq / 1000
            vol_vap = self.vol_tot - vol_liq

            mol_vap = pres_init * vol_vap / gas_ct / temp_init

            mol_vent = self.Vapor_1.moles - mol_vap

            mol_i = mol_liq * x_init + mol_vap * y_init
            mol_tot = mol_liq + mol_vap

            if self.oper_mode == 'Semibatch':
                Inlet = LiquidStream(self.paths, **self.inlet_inert_dict)
                Inlet.DynamicInlet = self.Inlet.DynamicInlet

                self.Inlet = Inlet

        else:
            temp_init, y_init = self.Liquid_1.getBubblePoint(pres=self.pres,
                                                             y_vap=True)

        # Moles of i
        mol_i = mol_liq * x_init + mol_vap * y_init

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

        hliq_init = self.Liquid_1.getEnthalpy(temp_init, basis='mole')
        hvap_init = self.Vapor_1.getEnthalpy(temp_init, basis='mole')

        diam = 0.438
        height_liq = vol_liq / (np.pi/4 * diam**2)
        area_ht = np.pi * diam * height_liq  # m**2

        temp_ht = self.Utility.get_inputs(0)['temp_in']

        ht_init = -self.h_conv * area_ht * (temp_init - temp_ht)

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
        sdot_init[:self.num_species + self.include_nitrogen] = dm_init
        sdot_init[-2] = du_init

        return states_init, sdot_init

    def __state_event(self, time, states, sdot, switch):
        events = []

        states_trim = np.split(states, self.trim_idx)
        sdot_trim = np.split(sdot, self.trim_idx)

        dict_states = dict(zip(self.name_states, states_trim))
        dict_sdot = dict(zip(self.name_states, sdot_trim))

        if any(switch):

            for di in self.state_events:
                if 'state_fn' in di.keys():
                    event_flag = di['state_fn'](time, self.Phases,
                                                dict_states, dict_sdot)
                else:
                    state_name = di['state_name']
                    state_idx = di['state_idx']
                    ref_value = di['value']

                    checked_value = dict_states[state_name][state_idx]

                    event_flag = ref_value - checked_value

                events.append(event_flag)

            # Maximum liquid constraint
            rho_liq = self.Liquid_1.getDensity(
                mole_frac=dict_states['x_liq'],
                temp=dict_states['temp'][0],
                basis='mole')

            vol_liq = dict_states['mol_liq'][0] / rho_liq / 1000  # m**3

            events.append(0.95 * self.vol_tot - vol_liq)

        return np.array(events)

    def __handle_event(self, solver, event_info):
        # pass
        state_event = event_info[0]

        for ind, val in enumerate(state_event[:-1]):
            direction = self.state_events[ind].get('direction')
            terminate = False
            id_event = self.state_events[ind].get('event_name')
            if val:
                if direction is None:
                    terminate = True
                elif direction == val:
                    terminate = True

                if terminate:
                    if id_event is None:
                        print('State event %i was reached' % (ind + 1))
                    else:
                        print("State event '%s' was reached" % id_event)

                    raise TerminateSimulation

        if state_event[-1]:
            print('95%% of the available tank volume reached by the liquid '
                  'phase')

            if self.stop_at_maxvol:
                raise TerminateSimulation
            else:
                self.allow_flow = False

    def solve_unit(self, runtime, verbose=True, sundials_opts=None):
        """ Solve Evaporator model


        Parameters
        ----------
        runtime : float
            final time of the simulation routine.
        verbose : bool, optional
            if True, integrator statistics will be displayed after the model
            is solved. The default is True.
        sundials_opts : dict, optional
            options to be passed to SUNDIALS. For a list of available options,
            visit https://jmodelica.org/assimulo/DAE_IDA.html
            The default is None.

        Returns
        -------
        time : list
            list of time steps taken by the numerical integrator.
        states : numpy array
            array containing the solution of the model.


        """

        num_comp = len(self.Liquid_1.name_species)
        len_in = [num_comp, 1, 1]
        states_in_dict = dict(zip(self.names_states_in, len_in))

        self.states_in_dict = {'Inlet': states_in_dict}

        self.args_inputs = (self, self.num_species)

        states_init, sdot_init = self.init_unit()

        # ---------- Solve
        n_comp = self.num_species + self.include_nitrogen
        alg_map = np.concatenate((np.ones(n_comp),
                                  np.zeros(2*n_comp + 3),
                                  [1, 0])
                                 )

        self.alg_map = (1 - alg_map).astype(bool)

        # ---------- Count states
        len_states = [n_comp] * 3 + [1] * 5
        self.trim_idx = np.cumsum(len_states)[:-1]

        # ---------- Check supercritical components
        temp_init = states_init[-1]

        is_supercritic = temp_init > self.Liquid_1.t_crit
        self.is_supercritic = is_supercritic

        # ---------- Solve problem
        # Create problem
        switches = [True] * len(self.state_events) + [True]
        problem = Implicit_Problem(self.unit_model,
                                   states_init, sdot_init,
                                   t0=self.elapsed_time, sw0=switches)

        problem.state_events = self.__state_event
        problem.handle_event = self.__handle_event

        problem.algvar = alg_map
        # problem.jac = self.unit_jacobian

        # Set solver
        solver = IDA(problem)
        solver.make_consistent('IDA_YA_YDP_INIT')
        solver.suppress_alg = True

        if not verbose:
            solver.verbosity = 50

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        runtime += self.elapsed_time

        # Solve
        time, states, sdot = solver.simulate(runtime)

        self.allow_flow = True  # Restore value for further analysis

        self.retrieve_results(time, states)

        return time, states

    def retrieve_results(self, time, states):
        self.elapsed_time += time[-1]

        # ---------- Create result object
        dp = unpack_states(states, self.dim_states, self.name_states)

        dp['time'] = np.asarray(time)

        self.outputs = dp

        self.profiles_runs.append(dp)
        dp = self.flatten_states()

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        # ---------- Update phases
        self.Liquid_1.temp = dp['temp'][-1]
        self.Liquid_1.pres = dp['pres'][-1]

        xliq_update = dp['x_liq']
        if self.include_nitrogen:
            xliq_update = xliq_update[:, :-1]
            Liquid_1 = LiquidPhase(self.paths[0],
                                   temp=dp['temp'][-1],
                                   pres=dp['pres'][-1],
                                   moles=dp['mol_liq'][-1],
                                   mole_frac=xliq_update[-1])

            self.Phases = Liquid_1

        else:
            self.Liquid_1.updatePhase(mole_frac=xliq_update[-1],
                                      moles=dp['mol_liq'][-1],
                                      temp=dp['temp'][-1],
                                      pres=dp['pres'][-1])

        # Output info
        self.Outlet = self.Liquid_1

        # ---------- Calculate duties
        self.get_heat_duty(time, states)

    def flatten_states(self):
        out = flatten_states(self.profiles_runs)

        return out

    def get_heat_duty(self, time, states):
        # ---------- Heat balance
        # Heating duty
        heat_bce = np.zeros_like(time)
        flow_vap = np.zeros_like(time)
        h_liq = np.zeros_like(time)
        h_vap = np.zeros_like(time)

        for ind, row in enumerate(states):
            mass_bce, q_ht = self.unit_model(time[ind], row, None, False)

            heat_bce[ind] = q_ht
            flow_vap[ind] = mass_bce[0]

            x_liq = self.result.x_liq[ind]
            y_vap = self.result.y_vap[ind]

            # if self.include_nitrogen:
            #     x_liq = x_liq[:-1]
            #     y_vap = y_vap[:-1]

            temp_bubble = self.Liquid_1.getBubblePoint(
                pres=self.result.pres[ind], mole_frac=x_liq)

            h_liq[ind] = self.Liquid_1.getEnthalpy(
                temp=temp_bubble, mole_frac=x_liq,
                basis='mole')

            h_vap[ind] = self.Vapor_1.getEnthalpy(
                temp=self.result.temp[ind], mole_frac=y_vap,
                basis='mole')

        # Condensation duty
        heat_cond_prof = flow_vap * (h_liq - h_vap)

        self.heat_profile = np.column_stack((heat_bce, heat_cond_prof))
        self.heat_duty = trapezoidal_rule(time, self.heat_profile)

        self.duty_type = [0, 0]  # TODO: this should depend on operation T

    def plot_profiles(self, pick_comp=None, **fig_kwargs):
        """
        Convenience function to plot model solution. Dynamic profiles displayed
        by this funcion are x_liq vs t, y_vap vs t, T/P vs t
        and mol_liq/mol_vap vs t.

        Parameters
        ----------
        pick_comp : list of int, optional
            indexes of states to be plot. If None, all the states are plotted.
            The default is None.
        **fig_kwargs : keyword arguments
            keyword arguments to be passed to the construction of fig and
            axes object of matplotlib (plt.subplots(**kwargs)).
            Do not use nrows or ncols arguments, since the plot grid is already
            defined by PharmaPy

        Returns
        -------
        fig : TYPE
            figure object.
        ax : numpy array
            grid of axis objects.

        """

        # Fractions
        if pick_comp is None:
            states_plot = ('x_liq', 'y_vap', 'temp', 'pres', 'mol_liq',
                           'mol_vap')
        else:
            states_plot = (['x_liq', pick_comp], ['y_vap', pick_comp], 'temp',
                           'pres', 'mol_liq', 'mol_vap')

        ylabels = ('x_liq', 'y_vap', 'T', 'P', 'N_L', 'N_V')
        fig, ax = plot_function(self, states_plot, fig_map=(0, 1, 2, 2, 3, 3),
                                ylabels=ylabels,
                                nrows=2, ncols=2, **fig_kwargs)

        for axis in ax.flatten():
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        fig.text(0.5, 0, 'time (s)', ha='center')

        fig.tight_layout()

        return fig, ax


class ContinuousEvaporator:
    """
    Create a continuous evaporator object

    Parameters
    ----------
    vol_drum : float
        total drum volume [m**3].
    adiabatic : bool, optional
        if True, heat transfer will be disregarded from the energy balance.
        The default is False.
    pressure : TYPE, optional
        pressure setpoint [Pa] (actual pressure is computed by the
        evaporator model). The default is 101325.
    diam_out : float, optional
        diameter of the vapor outlet pipe [m]. The default is 2.54e-2.
    frac_liq : float, optional
        setpoint for the fraction of the total tank volume occupied by the
        liquid phase. The default is 0.5.
    k_liq : float, optional
        proportional control constant for liquid level control, which
        dictates output liquid mole flow (F_L), with
        F_L = k_liq * (v_drum * frac_liq - V_L(t)), being V_L(t) the liquid
        volume computed by the DAE system. The default is 100.
    k_vap : float, optional
        proportional control constant for pressure, which
        actual pressure (P) by changing output vapor molar flow (F_V), with
        F_V = k_vap * f(pressure - P). The default is 1.
    h_conv : float, optional
        convective heat transfer coefficient for the liquid phase
        [W/m**2/K]. The default is 1000.
    activity_model : str, optional
        model to be used for calculation activity coefficient in
        vapor-liquid equilibria. Choose one of 'ideal', 'UNIFAC' and
        'UNIQUAC' The default is 'ideal'.
    num_interp_points : int, optional
        Number of interpolation points to be used by the Interpolation
        module of PharmaPy to calculate inputs if the evaporator receives
        material from a dynamic upstream unit operation. The default is 3.
    state_events : list of dicts, optional
        list of dictionaries containing the specification of a state event.
        To learn about the structure of a state event, see the
        PharmaPy.Commons.eval_state_events documentation.
        The default is None.
    reflux_ratio : float, optional
        reflux ratio ranging from (0 - 1), which dictates the fraction
        of the vapor flow which is sent back to the unit, assuming total
        condensation. The default is 0.

    Returns
    -------
    A continuous vaporizer object.

    """
    def __init__(self, vol_drum, adiabatic=False,
                 pressure=101325, diam_out=2.54e-2, frac_liq=0.5,
                 k_liq=100, k_vap=1,
                 cv_gas=0.8,
                 h_conv=1000,
                 activity_model='ideal', num_interp_points=3, mult_flash=1,
                 state_events=None, reflux_ratio=0):
 

        self._Inlet = None
        self._Phases = None
        self._Utility = None

        self.vol_tot = vol_drum
        self.vol_liq_set = vol_drum * frac_liq

        # Geometry
        self.diam_tank = (4/np.pi * vol_drum)**(1/3)
        self.area_base = np.pi / 4 * self.diam_tank**2
        self.area_out = np.pi / 4 * diam_out**2

        # Control
        self.k_liq = k_liq
        self.k_vap = k_vap
        self.cv_gas = cv_gas

        self.vol_offset = 1

        self.adiabatic = adiabatic

        self.pres = pressure

        # Jacobians
        # self.jac_states = jacauto(self.unit_model, 1)
        # self.jac_sdot = jacauto(self.unit_model, 2)

        # Heat transfer
        self.h_conv = h_conv

        self.oper_mode = 'Continuous'

        self.is_continuous = True

        self.tau = None

        self.profiles_runs = []

        self.activity_model = activity_model
        self.num_interp_points = num_interp_points

        self.nomenclature()

        self.state_event_list = state_events

        self.reflux_ratio = reflux_ratio

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phase):
        path_comp = phase.path_data

        vapor = VaporStream(path_comp, pres=self.pres,
                            mole_frac=np.zeros(len(phase.name_species)),
                            check_input=False, verbose=False)

        self._Phases = [phase, vapor]
        classify_phases(self)

        self.Liquid_1 = phase

        self.num_species = len(self.Liquid_1.mole_frac)
        self.name_species = self.Liquid_1.name_species

        self.__original_phase__ = copy.deepcopy(self.Liquid_1)

        self.states_di = {
            'mol_i': {'index': self.name_species, 'units': 'mol',
                      'dim': len(self.name_species), 'type': 'diff'},
            'x_liq': {'index': self.name_species, 'units': '',
                      'dim': len(self.name_species), 'type': 'alg'},
            'y_vap': {'index': self.name_species, 'units': '',
                      'dim': len(self.name_species), 'type': 'alg'},
            'mol_liq': {'units': 'mol', 'dim': 1, 'type': 'alg'},
            'mol_vap': {'units': 'mol', 'dim': 1, 'type': 'alg'},
            'pres': {'units': 'Pa', 'dim': 1, 'type': 'alg'},
            'u_int': {'units': 'J', 'dim': 1, 'type': 'diff'},
            'temp': {'units': 'K', 'dim': 1, 'type': 'alg'},
            }

        self.fstates_di = {
            'flow_liq': {'units': 'mol/s', 'dim': 1},
            'flow_vap': {'units': 'mol/s', 'dim': 1},
            'vol_liq': {'units': 'm**3', 'dim': 1},
            'vol_vap': {'units': 'm**3', 'dim': 1},
            }

        self.name_states = list(self.states_di.keys())

        self.dim_states = [di['dim'] for di in self.states_di.values()]

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):  # Create an inlet with additional species (N2)

        self._Inlet = inlet
        self._Inlet.num_interpolation_points = self.num_interp_points

        num_comp = len(self.Inlet.name_species)
        len_in = [num_comp, 1, 1]
        states_in_dict = dict(zip(self.names_states_in, len_in))

        self.states_in_dict = {'Inlet': states_in_dict}

    @property
    def Utility(self):
        return self._Utility

    @Utility.setter
    def Utility(self, utility):
        self.u_ht = 1 / (1 / self.h_conv + 1 / utility.h_conv)
        self._Utility = utility

    def nomenclature(self):
        self.names_states_in = ['mole_frac', 'mole_flow', 'temp']
        self.names_states_out = ['mole_frac', 'mole_flow', 'temp']
        # self.name_states = ['moles_i', 'x_liq', 'y_vap', 'mol_liq', 'mol_vap',
        #                     'pres', 'U_internal', 'temp']

        self.name_differential = ['moles_i', 'U_internal']
        self.name_algebraic = ['x_liq', 'y_vap', 'mol_liq', 'mol_vap',
                               'pres', 'temp']

    def get_inputs(self, time):
        inputs = get_inputs_new(time, self.Inlet, self.states_in_dict)

        return inputs

    def get_volumes(self, temp, pres, x_i, mol_liq, mol_vap):
        rho_liq = self.Liquid_1.getDensity(mole_frac=x_i, temp=temp,
                                           basis='mole')  # mol/L

        vol_liq = mol_liq / rho_liq / 1000  # m**3
        vol_vap = mol_vap * gas_ct * temp / pres

        return vol_liq, vol_vap

    def get_mole_flows(self, temp, pres, x_i, y_i, mol_liq, mol_vap,
                       input_flow):
        # Volumes
        rho_liq = self.Liquid_1.getDensity(mole_frac=x_i, temp=temp,
                                           basis='mole')  # mol/L

        vol_liq = mol_liq / rho_liq / 1000  # m**3
        vol_vap = mol_vap * gas_ct * temp / pres

        mw_vap = np.dot(self.Vapor_1.mw, y_i.T)

        # Gas density
        rho_mol = pres / gas_ct / temp  # mol/m**3
        rho_gas = rho_mol * (mw_vap / 1000)  # kg/m**3

        delta_p = (pres - self.pres)
        vel_vap = np.sqrt(np.maximum(eps, 2 * delta_p/rho_gas))
        # g_vapor = 2 / rho_gas * delta_p / (np.sqrt(abs(delta_p) + eps))  # See Sahlodin
        # vel_vap = np.sqrt(np.maximum(0, g_vapor))

        flow_vap = rho_mol * self.area_out * vel_vap * self.cv_gas * self.k_vap

        flow_liq = self.k_liq * (vol_liq - self.vol_liq_set) + input_flow
        flow_liq = np.maximum(0, flow_liq - flow_vap)

        return vol_liq, vol_vap, flow_liq, flow_vap

    def material_balances(self, time, mol_i, x_liq, y_vap,
                          mol_liq, mol_vap, pres, u_int, temp, u_inputs,
                          flows_out=False):

        input_flow = u_inputs['mole_flow']
        input_fracs = u_inputs['mole_frac']

        vol_liq, vol_vap, flow_liq, flow_vap = self.get_mole_flows(
            temp, pres, x_liq, y_vap, mol_liq, mol_vap, input_flow)

        vol_eqn = vol_liq + vol_vap - self.vol_tot

        if flows_out:
            return flow_liq, flow_vap, vol_liq
        else:
            # Differential eqns
            dmoli_dt = input_flow * input_fracs - flow_liq * x_liq - \
                (1 - self.reflux_ratio) * flow_vap * y_vap

            # Algebraic eqns
            component_bce = mol_liq * x_liq + mol_vap * y_vap - mol_i
            global_bce = mol_liq + mol_vap - sum(mol_i)

            k_i = self.Liquid_1.getKeqVLE(temp, pres, x_liq,
                                          self.activity_model)
            equilibria = y_vap - k_i * x_liq

            p_sat = self.Liquid_1.AntoineEquation(temp=temp)
            pres_eqn = np.dot(x_liq, p_sat) - pres

            alg_balances = np.concatenate((component_bce,  # x_liq
                                           equilibria,  # y_vap
                                           np.array([global_bce]),  # M_L
                                           np.array([vol_eqn]),  # M_V
                                           np.array([pres_eqn])  # P
                                           )
                                          )

            return dmoli_dt, alg_balances, flow_liq, flow_vap, vol_liq

    def energy_balances(self, time, flow_liq, flow_vap, vol_liq, u_int, temp,
                        x_liq, y_vap, mol_i, mol_liq, mol_vap, pres,
                        u_inputs, heat_prof=False):

        input_flow = u_inputs['mole_flow']
        input_fracs = u_inputs['mole_frac']
        input_temp = u_inputs['temp']

        # Enthalpies
        h_in = self.Inlet.getEnthalpy(temp=input_temp, mole_frac=input_fracs,
                                      basis='mole')

        h_liq = self.Liquid_1.getEnthalpy(temp, mole_frac=x_liq, basis='mole')

        if self.reflux_ratio == 0:
            h_top = self.Vapor_1.getEnthalpy(temp, mole_frac=y_vap,
                                             basis='mole')
        else:
            temp_bubble = self.Liquid_1.getBubblePoint(pres, mole_frac=y_vap)
            h_top = self.Liquid_1.getEnthalpy(temp=temp_bubble,
                                              mole_frac=y_vap, basis='mole')

        # Heat transfer
        if self.adiabatic:
            heat_transfer = 0
        else:
            height_liq = vol_liq / (np.pi/4 * self.diam_tank**2)
            area_ht = np.pi * self.diam_tank * height_liq + self.area_base

            ht_controls = self.Utility.get_inputs(time)
            temp_ht = ht_controls['temp_in']

            heat_transfer = self.h_conv * area_ht * (temp - temp_ht)

            if self.reflux_ratio == 0:
                q_cond = 0
                h_vap = h_top
            else:
                h_vap = self.Vapor_1.getEnthalpy(temp, mole_frac=y_vap,
                                                 basis='mole')

                q_cond = flow_vap * (h_vap - h_top)

            heat_transfer += q_cond

        flow_term = input_flow * h_in - flow_liq * h_liq - \
            (1 - self.reflux_ratio) * flow_vap * h_top
        if heat_prof:
            return flow_term, heat_transfer
        else:
            # Compute balances
            duint_dt = flow_term - heat_transfer

            pv_term = pres * self.vol_tot
            internal_energy = mol_liq * h_liq + mol_vap * h_vap - pv_term - u_int

            out_energy = np.array([duint_dt, internal_energy])

            return out_energy

    def unit_model(self, time, states, states_dot, sw, params=None,
                   enrgy_bce=False):

        # Decompose states
        di_states = unpack_states(states, self.dim_states, self.name_states)

        # Inputs
        u_inputs = self.get_inputs(time)['Inlet']

        # Material balance
        material_bces = self.material_balances(time, **di_states,
                                               u_inputs=u_inputs)

        material_bce = np.concatenate(material_bces[:-3])
        flow_liq, flow_vap, vol_liq = material_bces[-3:]

        if enrgy_bce:
            energy_terms = self.energy_balances(time, flow_liq, flow_vap, vol_liq,
                                                **di_states,
                                                u_inputs=u_inputs,
                                                heat_prof=True)

            return (flow_liq, flow_vap, vol_liq), energy_terms
        else:
            # Energy balance
            energy_bce = self.energy_balances(time, flow_liq, flow_vap, vol_liq,
                                              **di_states, u_inputs=u_inputs)

            # Concatenate balances
            balances = np.hstack((material_bce, energy_bce))

            if states_dot is not None:
                # Decompose derivatives
                di_dot = unpack_states(states_dot, self.dim_states,
                                       self.name_states)

                dmolesi_dt = di_dot['mol_i']
                duint_dt = di_dot['u_int']

                balances[:self.num_species] = balances[:self.num_species] - \
                    dmolesi_dt
                balances[-2] = balances[-2] - duint_dt

            # Update output objects
            self.Liquid_1.temp = di_states['temp']
            self.Vapor_1.temp = di_states['temp']

            self.Liquid_1.mole_frac = di_states['x_liq']
            self.Vapor_1.mole_flow = di_states['y_vap']

            return balances

    def unit_jacobian(self, c, time, states, sdot, params=None):
        jac_states = self.jac_states(time, states, sdot, params)
        jac_sdot = self.jac_sdot(time, states, sdot, params)

        jac_system = jac_states + c * jac_sdot

        return jac_system

    def init_unit(self):
        temp_bubble_init, y_init = self.Liquid_1.getBubblePoint(pres=self.pres,
                                                                y_vap=True)
        pres_init = self.pres

        # Mole fractions  # TODO: equilibrium compositions?
        x_init = self.Liquid_1.mole_frac
        # x_seed = np.append(x_seed, 0)

        # y_ = np.zeros(self.num_species + 1)
        # y_seed[-1] = 1

        # Moles of phases
        mol_liq = self.Liquid_1.moles
        vol_liq = self.Liquid_1.vol

        vol_vap = self.vol_tot - vol_liq
        if vol_vap < 0:
            raise ValueError(r"Drum volume ({:.2f} m3) lower than the liquid "
                             r"volume ({:.2f} m3)".format(self.vol_tot,
                                                          vol_liq))

        mol_vap = self.pres * vol_vap / gas_ct / temp_bubble_init

        self.Vapor_1.updatePhase(moles=mol_vap, mole_frac=y_init)

        # Moles of i
        mol_i = mol_liq * x_init + mol_vap * y_init

        u_inlet = self.get_inputs(0)['Inlet']
        inlet_flow = u_inlet['mole_flow']

        # Liquid flow
        flow_liq = self.k_liq * (vol_liq - self.vol_liq_set) + inlet_flow
        flow_liq = np.maximum(eps, flow_liq)

        dm_init = inlet_flow * u_inlet['mole_frac'] - flow_liq * x_init

        # ---------- Energy balance states
        # Enthalpies
        hin_init = self.Inlet.getEnthalpy(basis='mole', temp=u_inlet['temp'])
        hliq_init = self.Liquid_1.getEnthalpy(temp_bubble_init, basis='mole')
        hvap_init = self.Vapor_1.getEnthalpy(temp_bubble_init, basis='mole',
                                             mole_frac=y_init)

        # Heat transfer
        if self.adiabatic:
            heat_transfer = 0
        else:
            height_liq = vol_liq / (np.pi/4 * self.diam_tank**2)
            area_ht = np.pi * self.diam_tank * height_liq + self.area_base

            ht_controls = self.Utility.get_inputs(0)
            temp_ht = ht_controls['temp_in']

            heat_transfer = self.h_conv * area_ht * (temp_bubble_init -
                                                     temp_ht)

        # Disregard vapor flow at the beginning (vapor phase is at P_0)
        du_init = inlet_flow * hin_init - flow_liq * hliq_init - heat_transfer  # bce - dU_dt

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

    def _get_tau(self):
        time_upstream = getattr(self.Inlet, 'time_upstream')
        if time_upstream is None:
            time_upstream = [0]

        inputs = self.get_inputs(time_upstream[-1])['Inlet']

        dens_inlet = self.Liquid_1.getDensity(mole_frac=inputs['mole_frac'],
                                              temp=inputs['temp'],
                                              basis='mole') * 1000  # mol/m**3

        volflow_in = inputs['mole_flow'] / dens_inlet
        tau = self.vol_liq_set / volflow_in

        self.tau = tau
        return tau

    def _eval_state_events(self, time, states, sdot, sw):

        events = eval_state_events(
            time, states, sw, self.len_states,
            self.name_states, self.state_event_list, sdot=sdot,
            discretized_model=False, state_map=self.state_map)

        print(events)

        return events

    def solve_unit(self, runtime, steady_state=False, verbose=True,
                   sundials_opts=None, any_event=True):
        """
        Solve ContinuousEvaporator model

        Parameters
        ----------
        runtime : float
            final time of the simulation routine.
        steady_state : bool, optional
            If True, a steady-state version of the model is solved. Otherwise,
            a dynamic model is solved. The default is False.
        verbose : bool, optional
            if True, integrator statistics will be displayed after the model
            is solved. The default is True.
        sundials_opts : dict, optional
            options to be passed to SUNDIALS. For a list of available options,
            visit https://jmodelica.org/assimulo/ODE_CVode.html.
            The default is None.
        any_event : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        If steady_state, then a SciPy OptimizationResult object is returned.
        Else, a tuple containing a list of times returned by the numerical
        integrator and an array of solved states is returned.

        """

        self.args_inputs = (self, self.num_species)

        states_initial, sdev_initial = self.init_unit()

        # ---------- Solve
        n_comp = self.num_species
        alg_map = np.concatenate((np.ones(n_comp),
                                  np.zeros(2*n_comp + 3),
                                  [1, 0])
                                 )
        # dM_i/dt, (x_i, y_i), (P, M_L, M_V), (dU/dt, T)
        state_map = [1] + [0] * 2 + [0] * 3 + [1, 0]
        self.state_map = state_map

        # ---------- Solve problem
        if steady_state:
            def obj_fn(states): return self.unit_model(0, states)
            steady_solution = fsolve(obj_fn, states_initial)

            return steady_solution
        else:
            if self.state_event_list is None:
                def model(t, y, ydot, params=None, energy=False):
                    return self.unit_model(t, y, ydot, None, params, energy)
                problem = Implicit_Problem(model,
                                           states_initial, sdev_initial,
                                           t0=0)
            else:
                switches = [True] * len(self.state_event_list)

                def model(t, y, ydot, sw):
                    return self.unit_model(t, y, ydot, sw, None, False)

                problem = Implicit_Problem(model,
                                           states_initial, sdev_initial,
                                           t0=0, sw0=switches)

                def new_handle(solver, info):
                    return handle_events(
                        solver, info, self.state_event_list,
                        any_event=any_event)

                problem.state_events = self._eval_state_events
                problem.handle_event = new_handle

            problem.algvar = alg_map
            self.alg_map = alg_map
            # problem.jac = self.unit_jacobian

            # See self.name_states
            self.len_states = [self.num_species] * 3 + [1] * 5

            # Set solver
            solver = IDA(problem)
            solver.make_consistent('IDA_YA_YDP_INIT')
            solver.suppress_alg = True
            solver.report_continuously = True

            if sundials_opts is not None:
                for name, val in sundials_opts.items():
                    setattr(solver, name, val)

                    if name == 'time_limit':
                        solver.report_continuously = True

            if not verbose:
                solver.verbosity = 50

            if sundials_opts is not None:
                for name, val in sundials_opts.items():
                    setattr(solver, name, val)

            # Solve
            time, states, sdot = solver.simulate(runtime)

            self.retrieve_results(time, states)

            return time, states

    def retrieve_results(self, time, states):
        # self.elapsed_time += time[-1]

        # ---------- Create result object
        time = np.asarray(time)

        dp = unpack_states(states, self.dim_states, self.name_states)
        dp['time'] = time

        inputs_all = self.get_inputs(time)['Inlet']

        vol_liq, vol_vap, flow_liq, flow_vap = self.get_mole_flows(
            dp['temp'], dp['pres'], dp['x_liq'], dp['y_vap'], dp['mol_liq'],
            dp['mol_vap'], inputs_all['mole_flow'])

        dp['vol_liq'] = vol_liq
        dp['vol_vap'] = vol_vap

        dp['flow_liq'] = flow_liq
        dp['flow_vap'] = flow_vap

        # For connectivity purposes (what if the desired stream is vapor?)
        dp['mole_frac'] = dp['x_liq']
        dp['mole_flow'] = dp['flow_liq']

        self.profiles_runs.append(dp)

        dp = self.flatten_states()

        self.outputs = dp

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        # ---------- Update phases
        self.Liquid_1.temp = dp['temp'][-1]
        self.Liquid_1.pres = dp['pres'][-1]
        self.Liquid_1.updatePhase(mole_frac=dp['x_liq'][-1],
                                  moles=dp['mol_liq'][-1])

        holder = copy.deepcopy(self.__original_phase__)
        self.Phases = self.Liquid_1
        self.__original_phase__ = holder

        # ---------- Output info
        self.Outlet = LiquidStream(self.Liquid_1.path_data,
                                   temp=dp['temp'][-1],
                                   pres=dp['pres'][-1],
                                   mole_frac=dp['x_liq'][-1],
                                   mole_flow=flow_liq[-1])

        # ---------- Heat duties
        self.get_heat_duty(time, states)

    def get_heat_duty(self, time, states):
        heat_bce = np.zeros(len(time))
        h_liq = np.zeros_like(heat_bce)
        h_vap = np.zeros_like(heat_bce)

        flow_liq = np.zeros_like(heat_bce)
        flow_vap = np.zeros_like(heat_bce)

        for ind, row in enumerate(states):
            mat, energy = self.unit_model(time[ind], row,
                                          states_dot=None, sw=None,
                                          enrgy_bce=True)
            heat_bce[ind] = energy[1]

            temp_bubble = self.Liquid_1.getBubblePoint(
                pres=self.result.pres[ind],
                mole_frac=self.result.x_liq[ind])

            h_liq[ind] = self.Liquid_1.getEnthalpy(
                temp=temp_bubble,
                mole_frac=self.result.x_liq[ind], basis='mole')

            h_vap[ind] = self.Vapor_1.getEnthalpy(
                temp=self.result.temp[ind],
                mole_frac=self.result.y_vap[ind], basis='mole')

            flow_liq[ind] = mat[0]
            flow_vap[ind] = mat[1]

        # Condensation duty
        heat_cond_prof = flow_vap * (h_liq - h_vap)

        self.heat_profile = np.column_stack((heat_bce, heat_cond_prof))
        self.heat_duty = np.abs(trapezoidal_rule(time, self.heat_profile))
        self.duty_type = [0, 0]  # both are cooling water

        self.liqFlowProf = flow_liq
        self.vapFlowProf = flow_vap

    def flatten_states(self):
        out = flatten_states(self.profiles_runs)
        return out

    def plot_profiles(self, pick_comp=None, vol_plot=False, **fig_kwargs):
        """
        Convenience function to plot model solution. Dynamic profiles displayed
        by this funcion are x_liq vs t, y_vap vs t, T/P vs t and
        either mol_liq-mol_vap vs t or vol_liq-vol_vap vs t.

        Parameters
        ----------
        pick_comp : list of int, optional
            indexes of states to be plot. If None, all the states are plotted.
            The default is None.
        vol_plot : bool, optional
            If True, vol_liq-vol_vap vs t is plotted. Otherwise,
            mol_liq-mol_vap vs t is plotted.
        **fig_kwargs : keyword arguments
            keyword arguments to be passed to the construction of fig and
            axes objects of matplotlib (plt.subplots(**kwargs)).
            Do not use nrows or ncols arguments, since the plot grid is already
            defined by PharmaPy

        Returns
        -------
        fig : TYPE
            figure object.
        ax : numpy array
            grid of axis objects.

        """

        self.flatten_states()

        if pick_comp is None:
            pick_comp = np.arange(self.num_species)
        else:
            pick_comp = pick_comp

        # Fractions
        if pick_comp is None:
            states_plot = ['x_liq', 'y_vap', 'temp', 'pres']
        else:
            states_plot = [['x_liq', pick_comp], ['y_vap', pick_comp], 'temp',
                           'pres']

        ylabels = ['x_liq', 'y_vap', 'T', 'P']
        if vol_plot:
            states_plot += ['vol_liq', 'vol_vap']
            ylabels += ['V_L', 'V_V']
        else:
            states_plot += ['mol_liq', 'mol_vap']
            ylabels += ['N_L', 'N_V']

        fig, ax = plot_function(self, states_plot, fig_map=(0, 1, 2, 2, 3, 3),
                                ylabels=ylabels,
                                nrows=2, ncols=2, **fig_kwargs)

        for axis in ax.flatten():
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        fig.text(0.5, 0, 'time (s)', ha='center')

        fig.tight_layout()

        return fig, ax
