# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 00:24:18 2020

@author: huri
"""
import numpy as np
import numpy.matlib
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import CubicSpline

from PharmaPy.Phases import classify_phases
from PharmaPy.MixedPhases import Cake
from PharmaPy.SolidLiquidSep import high_resolution_fvm, get_sat_inf, upwind_fvm
from PharmaPy.NameAnalysis import get_dict_states
# from PharmaPy.Interpolation import SplineInterpolation
from PharmaPy.general_interpolation import define_initial_state
from PharmaPy.Commons import reorder_pde_outputs, eval_state_events, handle_events, unpack_discretized
from PharmaPy.Connections import get_inputs_new
# from pathlib import Path
from PharmaPy.Results import DynamicResult
from PharmaPy.Plotting import plot_distrib

eps = np.finfo(float).eps

gas_ct = 8.314


class Drying:
    def __init__(self, number_nodes, supercrit_names, diam_unit=0.01,
                 resist_medium=2.22e9, eta_fun=None, mass_eta=False,
                 state_events=None):
        """

        Parameters
        ----------
        number_nodes : float
            Number of apatial discretization of the cake along the axial
            coordinate.
        supercrit_names : list/tuple of str
            Names of the species in drying gas medium which doesn't participate
            in the mass transfer phenomenon. Names correspond to species names
            in the physical properties .json file.
        diam_unit : float (optional, default=0.01)
            Diameter of the dryer's cross section [m]
        resist_medium : float (optional, default=2.22e9)
            Mesh resistance of filter in dryer. [m**-1]
        eta_fun : callable
            Function with signature eta_fun(saturation). The default is None.
        mass_eta : bool (optional, default=None)
            If true, drying rate limiting factor is a function of mass fractional
            saturation value.
        state_events : list of dict (optional, default=None)
            Dictionary with keys respresenting properties used to monitor the event.
            Keys are composed of 'state_name', 'state_idx', 'value', 'callable'.
            Refer to 'parameter estimation' module for details.

        """
        self.supercrit_names = supercrit_names

        self.num_nodes = number_nodes
        self.station_diameter = diam_unit
        self.area_cross = self.station_diameter**2 * np.pi/4
        self.resist_medium = resist_medium
        self.T_ambient = 298

        # Transfer coefficients
        self.k_y = 1e-3  # mol/s/m**2 (Seader, Separation process)
        # self.h_T_j = 30  # W/m**2/K
        self.h_T_j = 10  # W/m**2/K
        self.h_T_loss = 30

        self._Phases = None
        self._Inlet = None

        # Limiting factor
        if eta_fun is None:
            eta_fun = lambda sat, mass_frac: 1

        self.eta_fun = eta_fun
        self.mass_eta = mass_eta
        self.oper_mode = 'Batch'
        self.is_continuous = False
        self.state_event_list = state_events

        self.outputs = None
        self.elapsed_time = 0

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if phases.__class__.__name__ == 'Cake':
            self.CakePhase = phases
            if self._Phases is None:
                self._Phases = phases.Phases
            else:
                self._Phases += phases.Phases
        elif phases.__module__ == 'PharmaPy.Phases':
            if self._Phases is None:
                self._Phases = [phases]
            else:
                self._Phases.append(phases)
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')

        if len(self._Phases) > 1:
            classify_phases(self)  # Enumerate phases: Liquid_1,..., Solid_1, ...
            self.cake_height = self.CakePhase.cake_vol / self.area_cross

            # ---------- Discretization
            self.z_grid = np.linspace(0, self.cake_height, self.num_nodes + 1)
            self.z_centers = (self.z_grid[1:] + self.z_grid[:-1]) / 2

            self.dz = np.diff(self.z_grid)

            names = self.Liquid_1.name_species
            idx_supercrit = [names.index(sup_name)
                             for sup_name in self.supercrit_names]
            self.idx_supercrit = np.atleast_1d(idx_supercrit)

            self.name_species = self.Liquid_1.name_species

            self.nomenclature()

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):
        self._Inlet = inlet

    def nomenclature(self):
        self.names_states_in = ['temp', 'mass_frac']
        self.names_states_out = self.names_states_in

        self.name_states = ['saturation', 'y_gas', 'x_liq', 'temp_gas',
                            'temp_cond']

        index_z = list(range(self.num_nodes))

        name_liq = [name for name in self.name_species
                    if name not in self.supercrit_names]
        self.states_di = {
            'saturation': {# 'index': index_z,
                           'dim': 1,
                           'units': '', 'type': 'diff'},
            'y_gas': {'index': self.name_species,
                      'dim': len(self.name_species), 'units': '',
                      'type': 'diff'},
            'x_liq': {'index': name_liq, 'dim': len(name_liq), 'units': '',
                      'type': 'diff'},
            'temp_gas': {# 'index': index_z,
                         'dim': 1, 'units': 'K',
                         'type': 'diff'},
            'temp_cond': {# 'index': index_z,
                         'dim': 1, 'units': 'K',
                         'type': 'diff'},
            }

        self.dim_states = [di['dim'] for di in self.states_di.values()]

        self.fstates_di = {}

    def get_inputs(self, time):

        input_dict = get_inputs_new(time, self.Inlet, self.states_in_dict)

        return input_dict

    def _eval_state_events(self, time, states, sw):

        events = eval_state_events(
            time, states, sw, self.len_states,
            self.name_states, self.state_event_list, sdot=self.derivatives,
            discretized_model=True)

        return events

    def get_y_equilib(self, temp_cond, x_liq, p_gas):
        mw_liq = self.Liquid_1.mw[self.idx_volatiles]
        x_liq_mole_frac = (x_liq / mw_liq).T / np.dot(1/mw_liq, x_liq.T)
        x_liq_mole_frac = x_liq_mole_frac.T

        p_sat = self.Liquid_1.AntoineEquation(temp=temp_cond)

        gamma = self.Liquid_1.getActivityCoeff(mole_frac=x_liq_mole_frac)
        # p_gas_total = np.sum(x_liq_mole_frac * p_sat[:, self.idx_volatiles],
        #                      axis=1)
        p_partial = (gamma * x_liq_mole_frac * p_sat[:, self.idx_volatiles]).T
        y_equil = p_partial  / p_gas

        return y_equil

    def get_drying_rate(self, x_liq, temp_cond, y_gas, p_gas):

        y_gas_mole_frac = self.Vapor_1.frac_to_frac(mass_frac=y_gas)
        y_equil = self.get_y_equilib(temp_cond, x_liq, p_gas)

        y_volat = y_gas_mole_frac[:, self.idx_volatiles].T  # * p_gas
        dry_volatiles = self.k_y * self.a_V * (y_equil - y_volat).T

        # dry_volatiles = self.k_y * (y_equil - y_volat).T
        dry_rates = np.zeros_like(y_gas_mole_frac)
        dry_rates[:, self.idx_volatiles] = dry_volatiles

        dry_rates[dry_rates < 0] = 0

        return dry_rates

    def unit_model(self, time, states, sw=None):
        '''
        state vector in the order: S|w_gas|w_liq|Tg|Ts
        '''

        num_comp = self.Liquid_1.num_species
        states_reord = states.reshape(-1, 3 + num_comp + self.num_volatiles)

        satur = states_reord[:, 0]
        y_gas = states_reord[:, 1:1 + num_comp]
        x_liq = states_reord[:, 1 + num_comp:
                             1 + num_comp + self.num_volatiles]
        x_liq[:, -2] = 0
        temp_gas = states_reord[:, -2]
        temp_sol = states_reord[:, -1]

        # ---------- Darcy's equation
        visc_gas = self.Vapor_1.getViscosity(temp=temp_gas,
                                             mass_frac=y_gas)

        sat_red = (satur - self.s_inf) / (1 - self.s_inf)
        sat_red = np.maximum(0, sat_red)
        k_ra = (1 - sat_red)**2 * (1 - sat_red**1.4)
        # vel_gas = self.k_perm * k_ra * self.dPg_dz / visc_gas
        vel_gas = self.dPg_dz/ \
        (self.CakePhase.alpha * visc_gas * self.rho_sol * (1 - self.porosity))/ np.mean(satur)
        
        # ---------- Drying rate term
        mw_avg_gas = np.dot(y_gas, self.Vapor_1.mw)
        rho_gas = self.pres_gas / gas_ct / temp_gas * mw_avg_gas / 1000# kg/m**3
        rho_liq_ = self.Liquid_1.rho_liq[self.idx_volatiles]
        self.rho_liq =  1 / np.sum((x_liq/ rho_liq_), axis=1)
        # Dry correction
        if self.mass_eta:
            rho_liq = self.rho_liq
            
            sat_eta = (self.porosity * satur * rho_liq)/ \
                ((1 - self.porosity) * self.rho_sol + self.porosity * satur *rho_liq)
            # sat_eta = satur * rho_liq / (satur*rho_liq + (1 - satur)*rho_gas)
            w_eta = x_liq
        else:
            sat_eta = satur
            w_eta = x_liq

        limiter_factor = self.eta_fun(sat_eta)#, w_eta)

        # Dry rate
        self.dry_rate = self.get_drying_rate(x_liq, temp_sol, y_gas,
                                             self.pres_gas)

        self.dry_rate *= limiter_factor[..., np.newaxis]

        # ---------- Model equations
        inputs = self.get_inputs(time)['Inlet']

        material_eqns = self.material_balance(
            time, satur, temp_gas, temp_sol, y_gas, x_liq,
            vel_gas, rho_gas, self.dry_rate, inputs)

        energy_eqns = self.energy_balance(time, temp_gas, temp_sol,
                                          satur, y_gas, x_liq, vel_gas,
                                          rho_gas, self.dry_rate, inputs)

        # print(satur.min())
        # print(inputs.values())

        model_eqns = np.column_stack(material_eqns + energy_eqns)

        self.derivatives = model_eqns.ravel()

        return model_eqns.ravel()

    def material_balance(self, time, satur, temp_gas, temp_sol, y_gas, x_liq,
                         u_gas, dens_gas, dry_rate, inputs, return_terms=False):
        
        satur[satur < eps] = eps
        satur[satur >= 1] = 1 - eps
        # ----- Reading inputs
        y_gas_inputs = inputs['mass_frac']

        # ----- Liquid phase
        dens_liq = self.rho_liq
        
        sum_dry = dry_rate.sum(axis=1)
        # sum_dry[sum_dry < eps] = 0
        
        dsat_dt = -sum_dry / dens_liq / self.porosity
        
        dxliq_dt = -1 / satur* \
            (dry_rate.T[self.idx_volatiles] / dens_liq / self.porosity +
              x_liq.T * dsat_dt)
            
        # dxliq_dt = np.zeros([len(self.idx_volatiles), len(satur)])
        
        # for ind, val in enumerate(sum_dry):
        #     if val == 0:
        #         dxliq_dt[self.idx_volatiles, ind] = 0
            
        #     else:
        #         dxliq_dt[self.idx_volatiles, ind] = -1 / satur[ind] * \
        #     (dry_rate.T[self.idx_volatiles, ind] / dens_liq[ind] / self.porosity +
        #       x_liq.T[self.idx_volatiles, ind] * dsat_dt[ind])

        # ----- Gas phase
        # Convective term
        epsilon_gas = self.porosity * (1 - satur)
        epsilon_gas[epsilon_gas <= eps] = eps

        # fluxes_yg = high_resolution_fvm(y_gas, boundary_cond=y_gas_inputs)
        fluxes_yg = upwind_fvm(y_gas, boundary_cond=y_gas_inputs)

        dygas_dz = np.diff(fluxes_yg, axis=0).T / self.dz

        convection = -u_gas * dygas_dz / epsilon_gas
        # Transfer term
        transfer_gas = dry_rate.T / epsilon_gas / dens_gas/ (1 - satur)
        # Dynamic saturation correction term
        total_mass_correction = y_gas.T / (1 - satur) * dsat_dt

        dygas_dt = convection + transfer_gas + total_mass_correction

        if return_terms:    # TODO: check term by term in material balance down this line
            self.masstrans_comp = 1

            return self.masstrans_comp

        else:
            return [dsat_dt, dygas_dt.T, dxliq_dt.T]

    def energy_balance(self, time, temp_gas, temp_sol, satur, y_gas, x_liq,
                       u_gas, rho_gas, dry_rate, inputs, return_terms=False):

        # temp_ref = 298
        mw_avg_gas = np.dot(y_gas, self.Vapor_1.mw)
        # ----- Reading inputs
        temp_gas_inputs = inputs['temp']

        # ----- Gas phase equations
        cpg_mix = self.Vapor_1.getCp(temp=temp_gas, mass_frac=y_gas,
                                     basis='mass')
        cvg_mix = cpg_mix - gas_ct / mw_avg_gas * 1000  # J/kg K

        epsilon_gas = self.porosity * (1 - satur)
        denom_gas = cvg_mix * epsilon_gas * rho_gas

        latent_heat = self.Vapor_1.getHeatVaporization(temp_sol,
                                                       basis='mass')

        xliq_extended = np.column_stack((x_liq, np.zeros((x_liq.shape[0],
                                                          len(self.idx_supercrit)))))

        cpl_mix = self.Liquid_1.getCp(temp=temp_sol, mass_frac=xliq_extended,
                                      basis='mass')
        temp_wb = 22+273
        sensible_heat = cpg_mix * (temp_gas - temp_wb) * dry_rate.sum(axis=1)
        
        heat_transf = self.h_T_j * self.a_V * (temp_gas - temp_sol)
        drying_terms = rho_gas / self.rho_liq * cpg_mix
        # heat_loss = self.h_T_loss * self.a_V * (temp_gas - self.T_ambient)
        heat_loss = self.h_T_loss * self.cake_height * (2*np.pi*1.5/2/100) *(temp_gas - self.T_ambient)
        heat_loss = 0  # This line is for assumption of no heat loss
        fluxes_Tg = high_resolution_fvm(temp_gas,
                                        boundary_cond=temp_gas_inputs)
        
        # fluxes_Tg = upwind_fvm(temp_gas, boundary_cond=temp_gas_inputs)
        # sensible_heat = cpg_mix * np.diff(fluxes_Tg) * dry_rate.sum(axis=1)
        dTg_dz = np.diff(fluxes_Tg) / self.dz * epsilon_gas * rho_gas

        conv_term = -u_gas * dTg_dz * cpg_mix * rho_gas

        dTg_dt = (conv_term + sensible_heat - heat_transf - heat_loss) / denom_gas
        
        # Empty port
        # dTg_dt = -u_gas * dTg_dz + (-heat_loss) / denom_gas
        # dTg_dt =  (conv_term - heat_loss) / denom_gas

        # print(dTg_dt[0])

        # ----- Condensed phases equations
        dens_liq = self.rho_liq
        # heat_loss_cond = self.h_T_loss * self.a_V * (temp_sol - self.T_ambient)
        heat_loss_cond = self.h_T_loss * self.cake_height * (2*np.pi*1.5/2/100) *(temp_sol - self.T_ambient)
        heat_loss_cond = 0
        drying_terms = (dry_rate[:, self.idx_volatiles] * latent_heat * 2).sum(axis=1)
        denom_cond = self.rho_sol * (1 - self.porosity) * self.cp_sol + \
            self.porosity * satur * cpl_mix * dens_liq

        dTcond_dt = (-drying_terms + heat_transf - heat_loss_cond) / denom_cond
        
        ## ---- Trial for lumping both cond/gas phase into one
        # dTtotal_dt = (conv_term + sensible_heat - drying_terms - heat_loss_cond)/ (denom_gas + denom_cond)
        
        # dTg_dt, dTcond_dt = dTtotal_dt, dTtotal_dt
        
        if return_terms:
            self.convec_term = u_gas * dTg_dz
            self.drying = drying_terms/ denom_gas
            self.heat_cond = heat_transf/ denom_gas
            self.heat_loss_emp = heat_loss/ denom_gas

            return self.convec_term, self.drying, self.heat_cond, self.heat_loss_emp

        else:

            return [dTg_dt, dTcond_dt]

    def solve_unit(self, deltaP, runtime=None, time_grid=None, any_event=True,
                   verbose=True, sundials_opts=None):
        
        p_atm=101325
        # ---------- Initialization
        # Volatile components
        idx_liquid = np.arange(0, self.Liquid_1.num_species)
        idx_volatiles = [i for i in idx_liquid if i not in self.idx_supercrit]
        self.num_volatiles = len(idx_volatiles)
        self.idx_volatiles = idx_volatiles

        num_y_gas = self.num_volatiles + len(self.idx_supercrit)
        num_x_liq = self.num_volatiles
        self.len_states = [1, num_y_gas, num_x_liq, 1, 1]
        num_comp = self.Liquid_1.num_species

        len_states_in = [1, num_y_gas]
        states_in_dict = dict(zip(self.names_states_in, len_states_in))
        self.states_in_dict = {'Inlet': states_in_dict}

        # Molar fractions
        # y_gas_init = np.tile(self.Vapor_1.mole_frac, (self.num_nodes,1))
        # x_liq_init = self.CakePhase.Liquid_1.mole_frac[:, idx_volatiles]

        y_gas_init = self.Vapor_1.mass_frac
        x_liq_init = self.CakePhase.Liquid_1.mass_frac

        satur_init = self.CakePhase.saturation

        # Temperatures
        temp_cond_init = self.CakePhase.Solid_1.temp
        temp_gas_init = self.Vapor_1.temp
        z_cake = self.CakePhase.z_external  # For drying_script_inyoung
        # z_cake = self.CakePhase.z_external # This line for 2MSMPR_Filter.py

        if x_liq_init.ndim == 1:
            x_liq_init = x_liq_init[idx_volatiles]
            
            if len(satur_init) != 1:
                states_tuple = (y_gas_init, x_liq_init, temp_gas_init, temp_cond_init)

                states_stacked = np.hstack(states_tuple)
                state_tiled = np.tile(states_stacked, (self.num_nodes, 1))
                states_prev = np.column_stack((satur_init, state_tiled))
                
            else:
                
                states_tuple = (satur_init, y_gas_init, x_liq_init, temp_gas_init)
    
                states_stacked = np.hstack(states_tuple)
                states_prev = np.tile(states_stacked, (self.num_nodes, 1))

        else:
            x_liq_init = x_liq_init[:, idx_volatiles]
            if y_gas_init.ndim == 1:
                y_gas_init = np.tile(y_gas_init, (self.num_nodes, 1))
            if isinstance(temp_cond_init, float):
                temp_cond_init = np.ones_like(satur_init) * temp_cond_init
            if isinstance(temp_gas_init, float):
                temp_gas_init = np.ones_like(satur_init) * temp_gas_init
                # temp_gas_init = np.tile(temp_gas_init, (self.num_nodes,1))
                # temp_cond_init = np.tile(temp_cond_init, (self.num_nodes,1))

            states_stacked = np.column_stack(
                (satur_init, y_gas_init, x_liq_init, temp_gas_init,
                 temp_cond_init))

            interp_obj = CubicSpline(z_cake, states_stacked)
            states_prev = interp_obj(self.z_centers)

        # states_init = states_prev
        # Merge states and interpolate in the grid nodes
        # states_prev = np.column_stack((satur_init, y_gas_init, x_liq_init,
        #                                temp_gas_init, temp_cond_init))

        # states_init = define_initial_state(state=states_stacked, z_after=self.z_centers,
        #              z_before=z_cake, indexed_state=True)

        #z_cake = self.cake_height * self.CakePhase.z_external

        # Physical properties
        alpha = self.CakePhase.alpha
        rho_sol = self.Solid_1.getDensity()
        porosity = self.CakePhase.porosity

        xliq = states_prev[:, num_comp + 1: num_comp + 1 + self.num_volatiles]
        # xliq = states_init[:, num_comp + 1: num_comp + 1 + self.num_volatiles]

        xliq_init = np.zeros((self.num_nodes, num_comp))
        xliq_init[:, self.idx_volatiles] = xliq
        rho_liq = self.Liquid_1.getDensity(temp=temp_cond_init,
                                           mass_frac=xliq_init, basis='mass')
        surf_tens = self.Liquid_1.getSurfTension(temp=temp_cond_init,
                                                 mass_frac=xliq_init)

        self.k_perm = 1 / alpha / rho_sol / (1 - porosity)
        # self.rho_liq = rho_liq
        self.rho_sol = rho_sol
        self.porosity = porosity
        self.cp_sol = self.Solid_1.getCp()

        # Mass transfer
        moments = self.Solid_1.getMoments(mom_num=[0, 1, 2, 3, 4])
        sauter_diam = moments[1] / moments[0]  # m

        # self.a_V = 6 / sauter_diam  # m**2/m**3
        self.a_V = moments[2] * (1 - porosity) / moments[3]
        # Gas pressure
        # deltaP_media = deltaP*self.resist_medium / \
        #     (alpha*rho_sol*self.cake_height + self.resist_medium)

        deltaP_media = deltaP*self.resist_medium / \
            (alpha*rho_sol*(1 - porosity)*self.cake_height +
              self.resist_medium)
        deltaP -= deltaP_media
        self.deltaP = deltaP
        p_top = p_atm + deltaP

        self.dPg_dz = deltaP / self.cake_height
        self.pres_gas = np.linspace(p_top, p_top - deltaP,
                                    num=self.num_nodes)

        # Irreducible saturation
        x_csd = self.Solid_1.x_distrib #* 1e-6
        csd = self.Solid_1.distrib# * 1e6
        mom_zero = self.Solid_1.moments[0]

        # rholiq_mass = np.mean(rho_liq[0] * self.Liquid_1.mw_av[0])  # kg/m**3
        self.s_inf = get_sat_inf(x_csd, csd, deltaP, porosity,
                                 self.cake_height, mom_zero,
                                 (np.mean(surf_tens), rho_liq[0]))#rholiq_mass))

        # ---------- Solve model
        model = self.unit_model
        if self.state_event_list is None:
            def model(t, y): return self.unit_model(t, y)#, None)
            problem = Explicit_Problem(self.unit_model, y0=states_prev.ravel(),
                                       t0=0)
        else:
            switches = [True] * len(self.state_event_list)
            problem = Explicit_Problem(self.unit_model, y0=states_prev.ravel(),
                                 t0=0, sw0=switches)

            def new_handle(solver, info):
                return handle_events(solver, info, self.state_event_list,
                                     any_event=any_event)

            problem.state_events = self._eval_state_events
            problem.handle_event = new_handle

        self.derivatives = model(0, states_prev.ravel())

        problem.name = 'Drying Model'

        sim = CVode(problem)

        if sundials_opts is not None:
            for key, val in sundials_opts.items():
                setattr(sim, key, val)

        # sim.linear_solver = 'SPGMR'
        
        if runtime is not None:
            final_time = runtime + self.elapsed_time

        if time_grid is not None:
            final_time = time_grid[-1] + self.elapsed_time
            self.elapsed_time = time_grid[0]
        
        if not verbose:
          sim.verbosity = 50
          
        time, states = sim.simulate(final_time,  ncp_list=time_grid)
        
        self.retrieve_results(time, states)

        return time, states

    def retrieve_results(self, time, states):
        time = np.array(time)
        self.timeProf = time
        self.elapsed_time += time[-1]
        
        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        inputs = self.get_inputs(time)['Inlet']

        dp = unpack_discretized(states, self.dim_states, self.name_states,
                                indexes=indexes, inputs=inputs)

        dp['time'] = time
        dp['z'] = self.z_centers
        
        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        self.CakePhase.z_external = self.z_centers

    def flatten_states(self):
        pass

    def plot_profiles(self, times=None, z_pos=None, pick_comp=None, **fig_kw):
        '''

        Parameters
        ----------
        time : int (optional, default=None)
            Integer value indicating the time on which calculated drying
            outputs to be plotted.
        z_pos : int (optional, default=None)
            Integer value indicating the axial position of cake coordniate
            at which calculated drying outputs to be plotted.
        pick_comp : list of lists (optional, default=None)
            Lists containing indexes/names of components to include in the
            x_liq and y_gas plots. First list contains indexes of compounds
            in liquid phase, and second list those of the gas phase.
            If None, all the existing components are plotted.

        '''
        if pick_comp is None:
            pick_liq = np.arange(self.num_liq)
            pick_vap = np.arange(self.num_gas)
        else:
            pick_liq, pick_vap = pick_comp

        if times is not None:

            states_plot = [('x_liq', pick_liq), ('y_gas', pick_vap),
                           'temp_cond', 'temp_gas', 'saturation']

            y_labels = ('x_liq', 'y_gas', 'T_cond', 'T_gas', 'sat')

            fig, axes = plot_distrib(self, states_plot, times=times,
                                     x_name='z', ncols=2, nrows=3,
                                     ylabels=y_labels, **fig_kw)

            fig.tight_layout()
            
        if z_pos is not None:
            
            states_plot = [('x_liq', pick_liq), ('y_gas', pick_vap),
                           'temp_cond', 'temp_gas', 'saturation']

            y_labels = ('x_liq', 'y_gas', 'T_cond', 'T_gas', 'sat')

            fig, axes = plot_distrib(self, states_plot, x_vals=z_pos,
                                     x_name='time', ncols=2, nrows=3,
                                     ylabels=y_labels, **fig_kw)

            fig.tight_layout()

        if z_pos is not None:  # TODO: this needs to be implemented
            fig, axes = plt.subplots(2, sharex=True, **fig_kw)

            idx_node = np.argmin(abs(z_pos - self.z_centers))
            w_liq = [self.xLiqProf[ind] for ind in pick_liq]
            w_vap = [self.yGasProf[ind] for ind in pick_vap]

            xliq_plot = np.vstack(w_liq)[:, idx_node].reshape(-1, len(self.timeProf))
            ygas_plot = np.vstack(w_vap)[:, idx_node].reshape(-1, len(self.timeProf))

            axes[0].plot(self.timeProf, xliq_plot.T)
            axes[1].plot(self.timeProf, ygas_plot.T)
            # axes[1].set_yscale('log')

            axes[0].text(1, 1.04, 'z_pos = %.3e m' % self.z_centers[idx_node],
                         ha='right', transform=axes[0].transAxes)

            axes[1].set_xlabel('$time$ (sec)')

            label_liq = [self.Liquid_1.name_species[ind] for ind in pick_liq]
            label_vap = [self.Liquid_1.name_species[ind] for ind in pick_vap]

            axes[0].legend(label_liq)
            axes[1].legend(label_vap)

            axes[0].set_ylabel('$x_{liq}$')
            axes[1].set_ylabel('$y_{gas}$')

        return fig, axes

    def plot_rates(self, z_pos=0, fig_size=None):
        z_idx = np.argmin(abs(z_pos - self.z_centers))
        temp_cond = self.tempLiqProf[:, z_idx]

        x_liq = np.column_stack([array[:, z_idx] for array in self.xLiqProf])
        y_gas = np.column_stack([array[:, z_idx] for array in self.yGasProf])

        gas_pressure = self.pres_gas[z_idx]

        self.dry_rate = self.get_drying_rate(x_liq, temp_cond, y_gas, gas_pressure)

        fig, axis = plt.subplots()

        axis.plot(self.timeProf, self.dry_rate)
        axis.set_xlabel('time (s)')
        axis.set_ylabel('drying rate ($\mathregular{mol_i \ s^{-1}}$)')

        axis.text(1, 1.04, '$z = %.2f$ m' % self.z_centers[z_idx], ha='right',
                  transform=axis.transAxes)

        axis.legend([self.Liquid_1.name_species[ind]
                     for ind in self.idx_volatiles], loc='best')

        return fig, axis
