# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:55:22 2020

@author: dcasasor
"""

import numpy as np
from scipy.optimize import newton

from PharmaPy.Commons import mid_fn
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


class EquilibriumLLE:
    def __init__(self, Inlet=None, temp_drum=None, pres_drum=None,
                 gamma_method='UNIQUAC'):
        # UNIQUAC

        self.temp = temp_drum
        self.pres = pres_drum
        self._Inlet = Inlet
        self.k_i = None

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

        if self.Inlet.__module__ == 'PharmaPy.Streams':
            self.in_flow = self.Inlet.mole_flow
        else:
            self.in_flow = self.Inlet.moles

    def material_eqn_based(self, extr, raff, x_extr, x_raff, z_i):

        gamma_extr = self.Inlet.getActivityCoeff(method=self.gamma_method,
                                                 mole_frac=x_extr,
                                                 temp=self.temp)

        gamma_raff = self.Inlet.getActivityCoeff(method=self.gamma_method,
                                                 mole_frac=x_raff,
                                                 temp=self.temp)

        global_bce = 1 - extr - raff
        comp_bces = z_i - extr*x_extr - raff*x_raff
        equilibria = x_extr * gamma_extr - x_raff * gamma_raff

        diff_frac = np.sum(x_extr - x_raff)
        args_mid = np.array([raff, diff_frac, raff - 1])
        vap_flow = mid_fn(args_mid)

        balance = np.concatenate(
            (np.array([global_bce]),
             comp_bces, equilibria,
             np.array([vap_flow]))
            )

        return balance

    def material_balance(self, phi_seed, x_1_seed, x_2_seed, z_i, temp):

        def get_ki(x1, x2, temp):
            gamma_1 = self.Inlet.getActivityCoeff(method=self.gamma_method,
                                                  mole_frac=x1,
                                                  temp=temp)

            gamma_2 = self.Inlet.getActivityCoeff(method=self.gamma_method,
                                                  mole_frac=x2,
                                                  temp=temp)

            k_i = gamma_1 / gamma_2

            return k_i

        def func_phi(phi, k_i):
            f_phi = z_i * (1 - k_i) / (1 + phi*(k_i - 1))

            return f_phi.sum()

        def deriv_phi(phi, k_i):
            deriv = z_i * (1 - k_i)**2 / (1 + phi*(k_i - 1))**2

            return deriv.sum()

        error = 1
        tol = 1e-4

        count = 0

        while error > tol:
            k_i = get_ki(x_1_seed, x_2_seed, self.temp)
            phi_k = newton(func_phi, phi_seed, args=(k_i, ), fprime=deriv_phi)

            x_1_k = z_i / (1 + phi_k*(k_i - 1))
            x_2_k = x_1_k * k_i

            # Normalize x's
            x_1_k *= 1 / x_1_k.sum()
            x_2_k *= 1 / x_2_k.sum()

            x_k = np.concatenate((x_1_k, x_2_k))
            x_km1 = np.concatenate((x_1_seed, x_2_seed))

            error = np.linalg.norm(x_k - x_km1)

            # Update
            x_1_seed = x_1_k
            x_2_seed = x_2_k
            phi_seed = phi_k

            count += 1

        # Retrieve results
        phi_conv = phi_seed
        x1_conv = x_1_seed
        x2_conv = x_2_seed

        info = {'error': error, 'num_iter': count}

        return phi_conv, x1_conv, x2_conv, info

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

    def unit_model(self, phi, x1, x2, temp, material=True):
        z_i = self.Inlet.mole_frac

        if material:
            balance = self.material_balance(phi, x1, x2, z_i, temp)
        else:
            balance = self.energy_balance(phi, x1, x2, z_i, temp)

        return balance

    def solve_unit(self):

        # Set seeds
        mol_z = self.in_flow * self.Inlet.mole_frac

        distr_seed = np.random.random(self.num_comp)
        distr_seed = distr_seed / distr_seed.sum()

        liq1_seed = mol_z * distr_seed
        liq2_seed = mol_z * (1 - distr_seed)

        x1_seed = liq1_seed / liq1_seed.sum()
        x2_seed = liq2_seed / liq2_seed.sum()

        phi_seed = liq2_seed.sum() / self.in_flow

        # Solve material balance
        solution = self.unit_model(phi_seed, x1_seed, x2_seed, self.temp)

        # # Energy balance
        # heat_bce = self.unit_model(solution, material=False)

        # Retrieve solution
        phase_part = solution[0]

        if phase_part > 1 or phase_part < 0:
            phase_part = 1
            print()
            print('No phases in equilibrium present at the specified conditions')
            print()

            xa_liq = self.Inlet.mole_frac
            xb_liq = xa_liq

        else:
            xa_liq = solution[1]
            xb_liq = solution[2]

        liquid_b = phase_part * self.in_flow
        liquid_a = self.in_flow - liquid_b

        path = self.Inlet.path_data

        # Store in objects
        if self.Inlet.__module__ == 'PharmaPy.Streams':
            Liquid_a = LiquidStream(path, mole_frac=xa_liq,
                                    mole_flow=liquid_a,
                                    temp=self.temp, pres=self.pres)
            Liquid_b = LiquidStream(path, mole_frac=xb_liq,
                                    mole_flow=liquid_b,
                                    temp=self.temp, pres=self.pres)
        else:
            Liquid_a = LiquidPhase(path, mole_frac=xa_liq,
                                   moles=liquid_a,
                                   temp=self.temp, pres=self.pres)
            Liquid_b = LiquidPhase(path, mole_frac=xb_liq,
                                   moles=liquid_b,
                                   temp=self.temp, pres=self.pres)

        dens_a = Liquid_a.getDensity()
        dens_b = Liquid_b.getDensity()

        if phase_part > 0 and phase_part < 1:
            if dens_a > dens_b:
                self.Liquid_2 = Liquid_a
                self.Liquid_3 = Liquid_b
            else:
                self.Liquid_2 = Liquid_b
                self.Liquid_3 = Liquid_a

        else:
            self.Liquid_2 = Liquid_b
            self.Liquid_3 = Liquid_a

        return solution
