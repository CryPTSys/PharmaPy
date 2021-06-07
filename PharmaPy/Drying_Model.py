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
from PharmaPy.Interpolation import SplineInterpolation
from PharmaPy.general_interpolation import define_initial_state
from PharmaPy.Commons import reorder_pde_outputs
# from pathlib import Path

eps = np.finfo(float).eps

gas_ct = 8.314


class Drying:
    def __init__(self, number_nodes, idx_supercrit, diam_unit=0.01,
                 resist_medium=2.22e9, eta_fun=None, mass_eta=False):
        """
        

        Parameters
        ----------
        number_nodes : TYPE
            DESCRIPTION.
        idx_supercrit : TYPE
            DESCRIPTION.
        diam_unit : TYPE, optional
            DESCRIPTION. The default is 0.01.
        resist_medium : TYPE, optional
            DESCRIPTION. The default is 2.22e9.
        eta_fun : TYPE, optional
            DESCRIPTION. The default is None.
        mass_eta : bool, optional
            If true, drying rate limiting factor is a function of mass fractional saturation value.
            The default is False.

        Returns
        -------
        None.

        """

        self.idx_supercrit = np.atleast_1d(idx_supercrit)

        self.num_nodes = number_nodes
        self.station_diameter = diam_unit
        self.area_cross = self.station_diameter**2 * np.pi/4
        self.resist_medium = resist_medium

        self.dP_media_vacuum = 3582.77
        self.T_ambient = 298

        # Transfer coefficients
        self.k_y = 1e-2  # mol/s/m**2 (Seader, Separation process)
        self.h_T_j = 30  # W/m**2/K
        self.h_T_j = 10  # W/m**2/K
        
        self.nomenclature()

        self._Phases = None
        self._Inlet = None

        # Limiting factor
        if eta_fun is None:
            eta_fun = lambda sat, mass_frac: 1

        self.eta_fun = eta_fun
        self.mass_eta = mass_eta
        self.oper_mode = 'Batch'
        self.is_continuous = False
        
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

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):
        self._Inlet = inlet

    def nomenclature(self):
        self.names_states_in = ['temp', 'mole_frac']
        self.names_states_out = self.names_states_in

        self.name_states = ['sat', 'y_gas', 'x_liq', 'temp_gas', 'temp_liq']

    def get_inputs(self, time):

        if self.Inlet.y_upstream is None or len(self.Inlet.y_upstream) == 1:
            input_dict = {'mole_frac': self.Inlet.mole_frac,
                          'temp': self.Inlet.temp}
        else:
            all_inputs = self.Inlet.InterpolateInputs(time)

            inputs = get_dict_states(self.names_upstream, self.num_concentr,
                                     0, all_inputs)

            input_dict = {}
            for name in self.names_states_in:
                input_dict[name] = inputs[self.bipartite[name]]

        return input_dict

    def get_drying_rate(self, x_liq, temp_cond, y_gas, p_gas):
        p_sat = self.Liquid_1.AntoineEquation(temp=temp_cond)

        gamma = self.Liquid_1.getActivityCoeff(mole_frac=x_liq)
        y_equil = (gamma * x_liq * p_sat[:, self.idx_volatiles]).T / p_gas
        
        y_volat = y_gas[:, self.idx_volatiles]
        dry_volatiles = self.k_y * self.a_V * (y_equil.T - y_volat)
        
        dry_rates = np.zeros_like(y_gas)
        dry_rates[:, self.idx_volatiles] = dry_volatiles

        return dry_rates

    def unit_model(self, time, states):
        '''
        state vector in the order: S|w_gas|w_liq|Tg|Ts
        '''

        num_comp = self.Liquid_1.num_species
        states_reord = states.reshape(-1, 3 + num_comp + self.num_volatiles)

        satur = states_reord[:, 0]
        y_gas = states_reord[:, 1:1 + num_comp]
        x_liq = states_reord[:, 1 + num_comp:
                             1 + num_comp + self.num_volatiles]
        temp_gas = states_reord[:, -2]
        temp_sol = states_reord[:, -1]

        # ---------- Darcy's equation
        visc_gas = self.Vapor_1.getViscosity(temp=temp_gas, mole_frac=y_gas)

        sat_red = (satur - self.s_inf) / (1 - self.s_inf)
        sat_red = np.maximum(0, sat_red)
        k_ra = (1 - sat_red)**2 * (1 - sat_red**1.4)
        vel_gas = self.k_perm * k_ra * self.dPg_dz / visc_gas

        # ---------- Drying rate term
        rho_gas = self.pres_gas / gas_ct / temp_gas  # mol/m**3

        # Dry correction
        if self.mass_eta:
            rho_liq = self.rho_liq
            sat_eta = satur * rho_liq / (satur*rho_liq + (1 - satur)*rho_gas)
            w_eta = x_liq
        else:
            sat_eta = satur
            w_eta = x_liq

        limiter_factor = self.eta_fun(sat_eta, w_eta)

        # Dry rate

        self.dry_rate = self.get_drying_rate(x_liq, temp_sol, y_gas, self.pres_gas)
        self.dry_rate *= limiter_factor[..., np.newaxis]
        
        # ---------- Model equations
        inputs = self.get_inputs(time)

        material_eqns = self.material_balance(
            time, satur, temp_gas, temp_sol, y_gas, x_liq,
            vel_gas, rho_gas, self.dry_rate, inputs)

        energy_eqns = self.energy_balance(time, temp_gas, temp_sol,
                                          satur, y_gas, x_liq, vel_gas,
                                          rho_gas, self.dry_rate, inputs)

        # print(satur.min())

        model_eqns = np.column_stack(material_eqns + energy_eqns)

        return model_eqns.ravel()

    def material_balance(self, time, satur, temp_gas, temp_sol, y_gas, x_liq,
                         u_gas, dens_gas, dry_rate, inputs, return_terms=False):

        # ----- Reading inputs
        y_gas_inputs = inputs['mole_frac']

        # ----- Liquid phase
        dens_liq = self.rho_liq * 1000  # mol/m**3
        dsat_dt = -dry_rate.sum(axis=1) / dens_liq / self.porosity

        dxliq_dt = -1 / satur * \
            (dry_rate.T[self.idx_volatiles] / dens_liq / self.porosity +
             x_liq.T * dsat_dt)

        # ----- Gas phase
        # Convective term
        epsilon_gas = self.porosity * (1 - satur)

        # fluxes_yg = high_resolution_fvm(y_gas, boundary_cond=y_gas_inputs)
        fluxes_yg = upwind_fvm(y_gas, boundary_cond=y_gas_inputs)

        dygas_dz = np.diff(fluxes_yg, axis=0).T / self.dz

        # Transfer term
        transfer_gas = dry_rate.T / epsilon_gas / dens_gas

        dygas_dt = -u_gas * dygas_dz + transfer_gas

        if return_terms:    # TODO: check term by term in material balance down this line
            self.masstrans_comp = 1

            return self.masstrans_comp

        else:
            return [dsat_dt, dygas_dt.T, dxliq_dt.T]

    def energy_balance(self, time, temp_gas, temp_sol, satur, y_gas, x_liq,
                       u_gas, rho_gas, dry_rate, inputs, return_terms=False):

        # ----- Reading inputs
        temp_gas_inputs = inputs['temp']

        # ----- Gas phase equations
        cpg_mix = self.Vapor_1.getCp(temp=temp_gas, mole_frac=y_gas,
                                     basis='mole')

        epsilon_gas = self.porosity * (1 - satur)
        denom_gas = cpg_mix * epsilon_gas * rho_gas

        heat_transf = self.h_T_j * self.a_V * (temp_gas - temp_sol)
        drying_terms = (dry_rate.T * cpg_mix * temp_gas).sum(axis=0)
        heat_loss = 14626.86 * (temp_gas - 295)
        # fluxes_Tg = high_resolution_fvm(temp_gas,
        #                                 boundary_cond=temp_gas_inputs)

        fluxes_Tg = upwind_fvm(temp_gas, boundary_cond=temp_gas_inputs)
        dTg_dz = np.diff(fluxes_Tg) / self.dz

        dTg_dt = -u_gas * dTg_dz + (drying_terms - heat_transf - heat_loss) / denom_gas

        # Empty port
        #dTg_dt = -u_gas * dTg_dz + (-heat_loss) / denom_gas

        # print(dTg_dt[0])

        # ----- Condensed phases equations
        dens_liq = self.rho_liq * 1000  # mol/m**3

        xliq_extended = np.column_stack((x_liq, np.zeros(x_liq.shape[0])))
        cpl_mix = self.Liquid_1.getCp(temp=temp_sol, mole_frac=xliq_extended,
                                      basis='mole')

        latent_heat = self.Vapor_1.getHeatVaporization(temp_sol,
                                                       idx=self.idx_volatiles,
                                                       basis='mole')

        denom_cond = self.rho_sol * (1 - self.porosity) * self.cp_sol + \
            self.porosity * satur * cpl_mix * dens_liq

        drying_terms_cond = (dry_rate[:, self.idx_volatiles] *
                             latent_heat).sum(axis=1)

        dTcond_dt = (-drying_terms_cond + heat_transf) / denom_cond


        if return_terms:
            self.convec_term = u_gas * dTg_dz
            self.drying = drying_terms/ denom_gas
            self.heat_cond = heat_transf/ denom_gas
            self.heat_loss_emp = heat_loss/ denom_gas

            return self.convec_term, self.drying, self.heat_cond, self.heat_loss_emp

        else:

            return [dTg_dt, dTcond_dt]

    def solve_unit(self, deltaP, runtime, p_atm=101325):

        # ---------- Discretization
        self.z_grid = np.linspace(0, self.cake_height, self.num_nodes + 1)
        self.z_centers = (self.z_grid[1:] + self.z_grid[:-1]) / 2

        self.dz = np.diff(self.z_grid)

        # ---------- Initialization
        # Volatile components
        idx_liquid = np.arange(0, self.Liquid_1.num_species)
        idx_volatiles = idx_liquid[idx_liquid != self.idx_supercrit]
        self.num_volatiles = len(idx_volatiles)
        self.idx_volatiles = idx_volatiles
        num_comp = self.Liquid_1.num_species

        # Molar fractions
        # y_gas_init = np.tile(self.Vapor_1.mole_frac, (self.num_nodes,1))
        # x_liq_init = self.CakePhase.Liquid_1.mole_frac[:, idx_volatiles]
        
        y_gas_init = self.Vapor_1.mole_frac
        x_liq_init = self.CakePhase.Liquid_1.mole_frac
        
        satur_init = self.CakePhase.saturation
        
        # Temperatures
        temp_cond_init = self.CakePhase.Solid_1.temp
        temp_gas_init = self.Vapor_1.temp
        z_cake = self.CakePhase.z_external # For drying_script_inyoung
        # z_cake = self.CakePhase.z_external # This line for 2MSMPR_Filter.py
        
        if x_liq_init.ndim == 1:
            x_liq_init = x_liq_init[idx_volatiles]
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
                                           mole_frac=xliq_init, basis='mole')
        surf_tens = self.Liquid_1.getSurfTension(temp=temp_cond_init,
                                                 mole_frac=xliq_init)

        self.k_perm = 1 / alpha / rho_sol / (1 - porosity)
        self.rho_liq = rho_liq
        self.rho_sol = rho_sol
        self.porosity = porosity
        self.cp_sol = self.Solid_1.getCp()

        # Mass transfer
        moments = self.Solid_1.getMoments(mom_num=[2, 3])
        sauter_diam = moments[1] / moments[0]  # m

        self.a_V = 6 / sauter_diam  # m**2/m**3
        
        # Gas pressure
        deltaP_media = deltaP*self.resist_medium / \
            (alpha*rho_sol*self.cake_height + self.resist_medium)
            
        # deltaP_media = deltaP*self.resist_medium / \
        #     (alpha*rho_sol*(1 - porosity)*self.cake_height +
        #      self.resist_medium)
        deltaP -= deltaP_media

        p_top = p_atm + deltaP

        self.dPg_dz = deltaP / self.cake_height
        self.pres_gas = np.linspace(p_top, p_top - deltaP,
                                    num=self.num_nodes)

        # Irreducible saturation
        x_csd = self.Solid_1.x_distrib * 1e-6
        csd = self.Solid_1.distrib * 1e6
        mom_zero = self.Solid_1.moments[0]

        rholiq_mass = np.mean(rho_liq[0] * self.Liquid_1.mw_av[0])  # kg/m**3
        self.s_inf = get_sat_inf(x_csd, csd, deltaP, porosity,
                                 self.cake_height, mom_zero,
                                 (np.mean(surf_tens), rholiq_mass))

        # ---------- Solve model
        model = Explicit_Problem(self.unit_model, y0=states_prev.ravel(),
                                 t0=0)

        model.name = 'Drying Model'
        sim = CVode(model)
        # sim.linear_solver = 'SPGMR'
        time, states = sim.simulate(runtime)

        self.retrieve_results(time, states)

        return time, states

    def retrieve_results(self, time, states):
        self.timeProf = np.array(time)

        num_gas = self.Vapor_1.num_species
        num_liq = self.num_volatiles

        sizes = [1, num_gas, num_liq, 1, 1]

        states_per_fv, states_reord = reorder_pde_outputs(
            states, self.num_nodes, sizes, name_states=self.name_states)

        self.SatProf = states_reord['sat']

        self.yGasProf = states_reord['y_gas']
        self.xLiqProf = states_reord['x_liq']

        self.tempGasProf = states_reord['temp_gas']
        self.tempLiqProf = states_reord['temp_liq']

        self.statesPerFiniteVolume = states_per_fv

        self.num_gas = num_gas
        self.num_liq = num_liq
    
    
    def flatten_states(self):
        pass
    
    
    def plot_profiles(self, time=None, z_pos=None, fig_size=None, jump=5,
                      pick_idx=None):
        if pick_idx is None:
            pick_liq = np.arange(self.num_liq)
            pick_vap = np.arange(self.num_gas)
        else:
            pick_liq, pick_vap = pick_idx

        if time is not None:
            fig, axes = plt.subplots(2, figsize=fig_size, sharex=True)

            idx_time = np.argmin(abs(time - self.timeProf))
            w_liq = [self.xLiqProf[ind] for ind in pick_liq]
            w_vap = [self.yGasProf[ind] for ind in pick_vap]

            xliq_plot = np.hstack(w_liq)[idx_time].reshape(-1, self.num_nodes)
            ygas_plot = np.hstack(w_vap)[idx_time].reshape(-1, self.num_nodes)

            axes[0].plot(self.z_centers, xliq_plot.T)
            axes[1].plot(self.z_centers, ygas_plot.T)

            axes[0].text(1, 1.04, 'time = %.3f s' % time, ha='right',
                         transform=axes[0].transAxes)

            axes[1].set_xlabel('$z$ (m)')

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
