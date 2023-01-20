# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:28:24 2020

@author: dcasasor
"""

import json
import numpy as np
import pandas as pd
import pathlib


def ParseDatabase(path_datafile, to_arrays=True):
    if isinstance(path_datafile, (list, tuple)):
        with open(path_datafile[0]) as file:
            original_data = json.load(file)

        for path in path_datafile[1:]:
            with open(path) as file:
                data = json.load(file)

            original_data.update(data)
    else:
        with open(path_datafile) as file:
            original_data = json.load(file)

    entries = []

    for key, dat in original_data.items():
        entries.append(dat.keys())

    # Extract interaction data, if existent
    interac = original_data.pop('interaction', None)

    # Collect data with the same key
    entries = set().union(*entries)

    dd = {}

    for entry in entries:
        vals = []
        tref_hvap = []
        for val in original_data.values():
            item = val.get(entry)
            if isinstance(item, dict):
                item = item['value']

                if entry == 'delta_hvap':
                    tref_hvap.append(val.get(entry)['temp_ref'])

            vals.append(item)

        dd[entry] = vals

        if len(tref_hvap) > 0:
            dd['tref_hvap'] = tref_hvap

    # Convert to arrays  # TODO: improve this
    if to_arrays:
        dd_arrays = {}
        for key, vals in dd.items():

            islist = [type(val) is list for val in vals]
            lengths = [len(val) for val in vals if isinstance(val, list)]

            if any(islist):  # there is multidimensional data
                length = max(lengths)
                props = []
                for val in vals:
                    if val is None:
                        props.append([0] * length)
                    else:
                        val_array = np.zeros(length)
                        val_array[:len(val)] = val
                        props.append(val_array)

            else:
                props = vals
            try:
                props = np.array(props, dtype=float)
            except:
                pass

            dd_arrays[key] = props

        dd_arrays['name_species'] = list(original_data.keys())

        if interac is not None:
            # Convert interaction parameters
            interac['amk'] = np.array(interac.get('amk', None))
            interac['vk'] = np.array(interac.get('vk', None))

            if 'unifac_groups' in interac:  # Avoid pandas xs warning
                interac['unifac_groups'] = [
                    tuple(pair) for pair in interac['unifac_groups']]

            dd_arrays.update(interac)
        return dd_arrays
    else:
        dd['name_species'] = list(original_data.keys())
        return dd


class ThermoPhysicalManager:
    def __init__(self, path_data):

        props_dict = ParseDatabase(path_data)
        self.__dict__ = props_dict
        self.name_species = props_dict['name_species']

        self.num_species = len(self.name_species)

        self.path_data = path_data

        # UNIFAC
        if 'unifac_groups' in props_dict:
            rk, qk, a_mat, b_mat, c_mat = self.get_UNIFACParams()
            self.Rk = rk
            self.Qk = qk
            self.a_unifac = a_mat
            self.b_unifac = b_mat
            self.c_unifac = c_mat

    def selectProperties(self, names):
        mapping = dict((key, ind)
                       for ind, key in enumerate(self.compound_names))

        idx_selection = [mapping[key] for key in names]

        for key, val in self.__dict__.items():
            values = getattr(self, key)
            if type(val) is list:
                vals = [values[i] for i in idx_selection]
                setattr(self, key, vals)
            else:
                vals = values[idx_selection]
                setattr(self, key, vals)

    def set_object(self):
        self.cpLiqPure = self.getCpLiqPure(temp=5)

    def getCpPure(self, temp, phase='liquid'):
        temp = np.atleast_1d(temp)
        num_temp = len(temp)

        if phase == 'liquid':
            cp_cts = np.atleast_2d(self.cp_liq)
        elif phase == 'solid':
            cp_cts = np.atleast_2d(self.cp_solid)
        elif phase == 'vapor':
            cp_cts = np.atleast_2d(self.cp_vapor)

        num_sp = len(cp_cts)
        ind_poly = np.arange(cp_cts.shape[1])

        cpMole = np.zeros((num_temp, num_sp))
        for ind, val in enumerate(temp):
            cpMole[ind] = np.dot(cp_cts, val**ind_poly)  # J/mol_j

        cpMass = cpMole / self.mw * 1000  # J/kg/K

        if len(cpMole) == 1:
            cpMole = cpMole[0]
            cpMass = cpMass[0]

        return cpMass, cpMole

    def getCpMix(self, temp, mass_frac=None, mole_frac=None, phase='liquid',
                 basis='mass'):
        # mass_frac = np.asarray(mass_frac)

        cp_mass, cp_mole = self.getCpPure(temp, phase=phase)

        if basis == 'mass':
            if mass_frac is None:
                mass_frac = self.frac_to_frac(mole_frac=mole_frac)

            if mass_frac.ndim == 1:
                cpMix = np.dot(mass_frac, cp_mass)
            elif mass_frac.ndim == 2:
                cpMix = (mass_frac * cp_mass).sum(axis=1)

        elif basis == 'mole':
            if mole_frac is None:
                mole_frac = self.frac_to_frac(mass_frac)

            if mole_frac.ndim == 1:
                cpMix = np.dot(mole_frac, cp_mole)
            elif mole_frac.ndim == 2:
                cpMix = (mole_frac * cp_mole).sum(axis=1)

        return cpMix

    def getEnthalpy(self, temp, temp_ref=298.15, mass_frac=None,
                    mole_frac=None, phase='liquid', basis='mass', idx=None,
                    total_h=True):
        temp = np.atleast_1d(temp)

        if idx is None:
            idx = np.arange(len(self.mw))

        if phase == 'liquid':
            cp_cts = np.atleast_2d(self.cp_liq)
        elif phase == 'solid':
            cp_cts = np.atleast_2d(self.cp_solid)
        elif phase == 'vapor':
            cp_cts = np.atleast_2d(self.cp_vapor)

        cp_cts = cp_cts[idx]

        ind_poly = np.arange(cp_cts.shape[1])
        exp = ind_poly + 1

        integral = []
        for ind, val in enumerate(temp):
            temp_term = val**exp - temp_ref**exp
            integral.append(np.dot(cp_cts / exp, temp_term))

        integral = np.vstack(integral)

        if total_h:
            if basis == 'mass':
                integralMass = integral * 1000 / self.mw[idx]  # J/kg_i
                if mass_frac is None:
                    mass_frac = self.frac_to_frac(mole_frac=mole_frac, ind=ind)

                if mass_frac.ndim == 1:
                    mass_fr = mass_frac[idx]
                else:
                    mass_fr = mass_frac[:, idx]

                enthalpyOut = (integralMass * mass_fr).sum(axis=1)
            elif basis == 'mole':
                if mole_frac is None:
                    mole_frac = self.frac_to_frac(mass_frac, ind=ind)

                if mole_frac.ndim == 1:
                    mole_fr = mole_frac[idx]
                else:
                    mole_fr = mole_frac[:, idx]

                # enthalpyOut = np.dot(integral, mole_fr.T)
                enthalpyOut = (integral * mole_fr).sum(axis=1)

            if len(enthalpyOut) == 1:
                enthalpyOut = enthalpyOut[0]

            return enthalpyOut
        else:
            if basis == 'mass':
                integralOut = integral * 1000 / self.mw[idx]  # J/kg_i
            else:
                integralOut = integral

            return integralOut

    def getHeatOfRxn(self, stoich_matrix, temp, mask, heat_rxn_ref, tref_hrxn):

        temp = np.atleast_1d(temp)
        temp_ref = tref_hrxn

        n_temp = len(temp)
        n_rxns, n_species = stoich_matrix.shape

        cp_cts = np.atleast_2d(self.cp_liq[mask])

        ind_poly = np.arange(cp_cts.shape[1])
        exp = ind_poly + 1

        integral = np.zeros((n_temp, n_species))
        for ind, val in enumerate(temp):
            temp_term = val**exp - temp_ref**exp
            integral[ind] = np.dot(cp_cts / exp, temp_term)  # J/mol_j

        delta_cp = np.dot(integral, stoich_matrix.T)
        heat_of_rxn = heat_rxn_ref + delta_cp

        if len(heat_of_rxn) == 1:
            heat_of_rxn = heat_of_rxn[0]

        return heat_of_rxn  # J/mol

    def getDensityPure(self, phase='liquid', temp=None):
        if phase == 'liquid':
            rhoMass = self.rho_liq  # TODO: T-dependent rho
        elif phase == 'solid':
            rhoMass = self.rho_solid

        rhoMole = rhoMass / self.mw  # kmol/m**3  (mol/L)

        return rhoMass, rhoMole

    def getDensityMix(self, mass_frac=None, mole_frac=None, phase='liquid',
                      temp=None, basis='mass'):

        if temp is None:
            temp = self.temp

        rhoMass, rhoMole = self.getDensityPure(phase, temp)

        if basis == 'mass':
            if mass_frac is None:
                mass_frac = self.frac_to_frac(mole_frac=mole_frac)

            rhoMix = 1 / np.dot(mass_frac, 1 / rhoMass)

        else:
            if mole_frac is None:
                mole_frac = self.frac_to_frac(mass_frac=mass_frac)

            rhoMix = 1 / np.dot(mole_frac, 1 / rhoMole)

        return rhoMix

    def getMolWeight(self, mole_frac=None, mass_frac=None):
        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac
        elif mass_frac is not None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)

        mw_av = np.dot(self.mw, mole_frac.T)

        return mw_av

    def getViscosityPure(self, phase='liquid', temp=None):
        if temp is None:
            temp = self.temp

        if phase == 'liquid':
            visc_cts = np.atleast_2d(self.visc_liq)
            temp_term = np.array([1, 1/temp, temp, temp**2])

            viscosity = 10**(np.dot(visc_cts, temp_term))/1000  # Pa*s

        elif phase == 'vapor':
            # to be impelemented
            viscosity = self.visc_gas

        return viscosity

    def getViscosityMix(self, temp=None, mass_frac=None, mole_frac=None,
                        phase='liquid'):

        if temp is None:
            temp = self.temp

        visc_comp = self.getViscosityPure(phase, temp)

        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac
        elif mole_frac is None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)

        # Mixing rules
        if phase == 'liquid':
            viscMix = np.exp(
                np.dot(mole_frac, np.log(visc_comp)))

        elif phase == 'vapor':
            if visc_comp.ndim == 1:
                visc_term = np.outer(visc_comp, 1/visc_comp)
                mw_term = np.outer(self.mw, 1/self.mw)

                phi_mix = (1 + visc_term**0.5 * mw_term**0.25)**2 / \
                    np.sqrt(8*(1 + mw_term))

                interactions = np.dot(mole_frac, phi_mix.T)

                viscMix = (visc_comp * mole_frac / interactions).sum(axis=1)

        return viscMix

    def getDiffusivityPure(self, wrt, temp=None):
        diffusivity = self.diffusivity[:, wrt]
       # diffusivity =  0.00442005
        return diffusivity

    def frac_to_conc(self, mass_frac=None, mole_frac=None, basis='mole'):
        densMass, densMole = self.getDensityPure()
        if mass_frac is None:
            if mole_frac.ndim == 1:
                concentr = mole_frac / np.dot(mole_frac, 1 / densMole)
            else:
                concentr = mole_frac.T / np.dot(mole_frac, 1 / densMole)
                concentr = concentr.T
        else:
            if mass_frac.ndim == 1:
                concentr = (mass_frac / self.mw) / np.dot(mass_frac,
                                                          1 / densMass)
            else:
                concentr = (mass_frac / self.mw).T / np.dot(mass_frac,
                                                            1 / densMass)
                concentr = concentr.T

        if basis == 'mass':
            concentr *= self.mw

        return concentr  # mol/L (kmol/m**3) - kg/m**3

    def frac_to_frac(self, mass_frac=None, mole_frac=None, ind=None):
        if mole_frac is not None:
            if mole_frac.ndim == 1:
                mass_frac = mole_frac * self.mw / np.dot(mole_frac, self.mw)
                frac_out = mass_frac
            else:
                mass_frac = (mole_frac * self.mw).T / np.dot(mole_frac, self.mw)
                frac_out = mass_frac.T
        else:
            if mass_frac.ndim == 1:
                mole_frac = (mass_frac / self.mw) / np.dot(mass_frac, 1/self.mw)
                frac_out = mole_frac
            else:
                mole_frac = (mass_frac / self.mw).T / np.dot(mass_frac, 1/self.mw)
                frac_out = mole_frac.T

        return frac_out

    def conc_to_frac(self, conc, solvent_ind=None, basis=None):
        conc = np.asarray(conc)

        if solvent_ind is not None:
            _, densMole = self.getDensityPure(phase='liquid')
            molVol = 1 / densMole  # mol/L, kmol/m3

            mask_solv = np.ones_like(conc, dtype=bool)
            mask_solv[solvent_ind] = False

            conc_solv = (1 - np.dot(conc[mask_solv], molVol[mask_solv])) / \
                molVol[solvent_ind]

            conc[solvent_ind] = conc_solv

        if conc.ndim == 1:
            mole_frac = conc / conc.sum()
        else:
            mole_frac = (conc.T / conc.sum(axis=1)).T

        if basis == 'mole':
            if solvent_ind:
                return mole_frac, conc
            else:
                return mole_frac

        elif basis == 'mass':
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)
            if solvent_ind:
                return mass_frac, conc
            else:
                return mass_frac
        else:
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)
            if solvent_ind:
                return mass_frac, mole_frac, conc
            else:
                return mass_frac, mole_frac

    def mass_conc_to_frac(self, conc, solvent_ind=None, basis=None):
        conc = np.asarray(conc)

        if solvent_ind is not None:
            dens_mass, _ = self.getDensityPure(phase='liquid')
            mask_solv = np.ones_like(conc, dtype=bool)
            mask_solv[solvent_ind] = False

            conc_solv = (
                1 - np.dot(conc[mask_solv], 1/dens_mass[mask_solv])) * \
                dens_mass[solvent_ind]

            conc[solvent_ind] = conc_solv

        if conc.ndim == 1:
            mass_frac = conc / conc.sum()
        else:
            mass_frac = (conc.T / conc.sum(axis=1)).T

        if basis == 'mass':
            if solvent_ind:
                return mass_frac, conc
            else:
                return mass_frac

        elif basis == 'mole':
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)
            if solvent_ind:
                return mole_frac, conc
            else:
                return mole_frac

        else:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)
            if solvent_ind is None:
                return mass_frac, mole_frac
            else:
                return mass_frac, mole_frac, conc

    def conc_to_conc(self, mass_conc=None, mole_conc=None):
        if mass_conc is not None:
            conv_conc = mass_conc / self.mw  # mol / L
        else:
            conv_conc = mole_conc * self.mw  # kg / m3

        return conv_conc

    def getMolVolMix(self, frac):
        molvolMix = np.dot(self.mol_vol, frac)

        return molvolMix

    def AntoineEquation(self, temp=None, pres=None):
        a_ct, b_ct, c_ct = self.p_vap.T

        if pres is None:
            if isinstance(temp, np.ndarray):
                temp = temp[..., np.newaxis]

            vap_pressure = a_ct - b_ct / (temp + c_ct)

            return 10**(vap_pressure)

        else:
            if isinstance(pres, np.ndarray):
                pres = pres[..., np.newaxis]

            temp_sat = b_ct / (a_ct - np.log10(pres)) - c_ct

            return temp_sat

    def getKeqVLE(self, temp=None, pres=None, x_liq=None, y_vap=None,
                  gamma_model='ideal'):

        if temp is None:
            temp = self.temp

        if pres is None:
            pres = self.pres

        if x_liq is None:
            x_liq = self.mole_frac

        crit = isinstance(temp, np.ndarray) and temp.ndim == 1
        if crit:
            p_vap = self.AntoineEquation(temp)
            supercrit = np.ones_like(p_vap) * temp[:, np.newaxis] > self.t_crit
            if np.any(supercrit):
                for row in supercrit:
                    p_vap[:, row] = self.henry_constant[row]
        else:
            supercrit = temp > self.t_crit
            p_vap = self.AntoineEquation(temp)
            if any(supercrit):
                p_vap[supercrit] = self.henry_constant[supercrit]

        if gamma_model == 'ideal':
            gamma = np.ones_like(x_liq)
        elif gamma_model == 'UNIFAC':
            gamma = self.UNIFAC_DMD(x_liq, temp)
        elif gamma_model == 'UNIQUAC':
            gamma = self.UNIQUAC(x_liq, temp)

        k_vals = p_vap * gamma / pres

        return k_vals

    def UNIQUAC(self, mole_frac=None, temp=None):
        r""" Calculate activity coefficients :math:`\gamma_i` using UNIQUAC model.

        Parameters
        ------------
        x_liq : ndarray
            liquid molar fractions for n components.
        temp : float
            temperature [K]
        amk : ndarray
            n x n array containing interaction parameters [J/mol]:
                component m (row) respect to component k (column):

            .. math::

               \begin{bmatrix}
                   a_{11}       & a_{12}          & \cdots    & a_{1 n_{comp}} \\
                   a_{21}       & a_{22}          & \cdots    & a_{2 n_{comp}} \\
                   \vdots       & \vdots          & \ddots    & \vdots \\
                   a_{n_{comp}} & a_{n_{comp} 2}  & \cdots    & a_{n_{comp} n_{comp}}
               \end{bmatrix}

        ri : ndarray
            molecular volume constants (n-sized array).
        qi : ndarray
            molecular surface area constants (n-sized array).
        qip : ndarray
            molecular surface area for systems containing water or alcohols
            (n-sized array). Usually, qi = qip.

        Returns
        -----------
        output : ndarray
            n-sized array with activity coefficients.

        Notes
        ---------------------------
        UNIQUAC model assumes that activity coefficient is the sum of a
        combinatorial and a residual part [1], [2], [3]:

        .. math::

            ln\gamma_i = ln\gamma_i^{comb} + ln\gamma_i^{res}, \, i = 1, \cdots, n

        with

        .. math::

            ln\gamma_i^{comb} = ln\frac{\phi_i (r_i, x_i)}{x_i} +
                                5q_i ln\frac{\theta_i(q_i, x_i)}{\phi_i} +
                                l_i (r_i, q_i) -
                                \frac{\phi_i}{x_i} \sum_j x_j l_j

        and

        .. math::

            ln\gamma_i^{res} =  q'_i \left[ 1 - ln \left( \sum_j \theta'_j (q'_j, x_j) \cdot \tau_{ji}(a_{ji}, T) \right) +
                                \sum_j \frac{\theta'_j\tau_{ij}}{\sum_k \theta'_k \tau_{kj}} \right]

        Temperature dependency is included in the term :math:`\tau_{ij}`:

        .. math::
            \tau_{ij} = exp \left[ \frac{-a_{mk}}{R T} \right]

        where :math:`a_{mk}` are *interaction* coefficients obtained by fitting
        experimental data.


        References
        ----------------
        [1] J. M. Smith, H. Van Ness and M. Abbott, Introduction to Chemical
        Engineering Thermodynamics, McGraw Hill, 7th Ed., 2004.

        [2] J. M. Prausnitz, R. N. Lichtenthaler and E. Gomes de Acevedo, Molecular
        thermodynamics of fluid-phase equilibria, Prentice Hall, New Jersey, 1999.

        [3] G. Kontogeorgis and G. Folas, Thermodynamic Models for Industrial
        Applications, John Willey & Sons, Inc., West Sussex, First Edit., 2010.

        """

        # Rename
        ri = self.ri
        qi = self.qi
        qip = self.qip
        amk = self.amk

        x_liq = mole_frac

        temp = np.asarray(temp)
        one_dimensional_output = False
    #    temp = temp[np.newaxis]

        if x_liq.ndim == 1:
            one_dimensional_output = True
            x_liq = x_liq[np.newaxis, :]
            temp = temp[np.newaxis]

        num_gammas, num_components = x_liq.shape

        # # Avoid log of zero
        # x_liq[x_liq == 0] = 1e-15

        gas_constant = 8.314  # J/mol/K

        # Avoid two-dimensional arrays
        ri = ri.ravel()
        qi = qi.ravel()
        qip = qip.ravel()

        # -------------------- Preliminary calculations
        tau = []
        for tp in temp:
            tau.append(np.exp(-amk / gas_constant / tp))

        phi = ri / (ri * x_liq).sum(axis=1)[:, np.newaxis]
        theta = qi / (qi * x_liq).sum(axis=1)[:, np.newaxis]

        li = 5*(ri - qi) - (ri - 1)

        # -------------------- Combinatorial term (same as non-modified UNIFAC)
        gamma_combinatorial = np.log(phi) + 5 * qi * np.log(theta / phi) +\
            li - phi * (li * x_liq).sum(axis=1)[:, np.newaxis]

        # -------------------- Residual term, with qip instead of qi
        theta_p = x_liq * qip / (qip * x_liq).sum(axis=1)[:, np.newaxis]

        tau_theta = np.zeros((num_gammas, num_components))
        for ind, row in enumerate(theta_p):
            tau_theta[ind] = (tau[ind].T * row).sum(axis=1)

        tau_theta_residual = np.zeros_like(tau_theta)
        for ind, row in enumerate(theta_p):
            tau_theta_residual[ind] = (tau[ind] * row / tau_theta[ind]).sum(axis=1)

        gamma_residual = qip*(1 - np.log(tau_theta) - tau_theta_residual)

        # -------------------- Unify terms
        log_gamma = gamma_residual + gamma_combinatorial

        gamma = np.exp(log_gamma)

        if one_dimensional_output:
            return np.squeeze(gamma)
        else:
            return gamma

    def get_UNIFACParams(self, dataframes=False):
        """ Get UNIFAC-Dortmund constants by specifying the indexes of the groups
            present in the mixture

            Parameters
            ----------
            group_idx : list of tuples
                list containing 2-tuples, which describe the main and secondary
                group index (without repetition) for each group in the mixture.

            Returns
            -------

            Example
            -------
                For a water ethanol mixture, group_idx would be:

                    >>> water_ethanol = [(7, 16), (1, 1), (1, 2), (5, 14)]

                The first tuple (7, 16) corresponds to water, and the next three
                conform ethanol: [CH3, CH2 and OH(p)]. For information on the
                group numbers, refer to [1]

            References
            ----------
            [1] Gmehling, J.; Li, J.; Schiller, M. Ind. Eng. Chem. Res. 1993,
            32 (1), 178–193.

        """
        group_tuples = self.unifac_groups
        group_idx = np.array(self.unifac_groups)

        # Import data
        root = str(pathlib.Path(__file__).parents[1]) + '/data/thermodynamics/'
        interac_path = root + 'unifac_interaction_params.csv'

        interac_data = pd.read_csv(interac_path, index_col=(0, 1))

        rq_path = root + 'unifac_rk_qk.csv'
        rk_qk = pd.read_csv(rq_path, index_col=(2, 0))

        # Create empty arrays
        num_groups = len(group_idx)
        a_matrix = np.zeros((num_groups, num_groups))
        b_matrix = np.zeros_like(a_matrix)
        c_matrix = np.zeros_like(a_matrix)

        main = group_idx[:, 0]

        # Create a grid with coordinates
        j_coord, i_coord = np.meshgrid(main, main)

        for m in range(num_groups):
            for n in range(num_groups):
                i, j = (i_coord[m, n], j_coord[m, n])

                if i == j:
                    pass
                elif i < j:
                    a_matrix[m, n] = interac_data.loc[i, j]['anm']
                    b_matrix[m, n] = interac_data.loc[i, j]['bnm']
                    c_matrix[m, n] = interac_data.loc[i, j]['cnm']
                else:
                    a_matrix[m, n] = interac_data.loc[j, i]['amn']
                    b_matrix[m, n] = interac_data.loc[j, i]['bmn']
                    c_matrix[m, n] = interac_data.loc[j, i]['cmn']

        r_k = []
        q_k = []
        for ind in group_tuples:
            r_k.append(rk_qk['Rk'].xs(ind))
            q_k.append(rk_qk['Qk'].xs(ind))

        r_k = np.array(r_k)
        q_k = np.array(q_k)

        if dataframes:
            a_matrix = pd.DataFrame(a_matrix, index=group_tuples,
                                    columns=group_tuples)
            b_matrix = pd.DataFrame(b_matrix, index=group_tuples,
                                    columns=group_tuples)
            c_matrix = pd.DataFrame(c_matrix, index=group_tuples,
                                    columns=group_tuples)

            r_k = pd.Series(r_k, index=group_tuples)
            q_k = pd.Series(q_k, index=group_tuples)

        return [r_k, q_k, a_matrix, b_matrix, c_matrix]

    def UNIFAC_DMD(self, x_i=None, temp=None):

        """ Calculate activity coefficients of a liquid mixture using the UNIFAC
        group-contribution method

        Parameters
        ----------
        x_i : array
            liquid molar fractions
        temp : float or array
            temperature (K)
        r_k : array
            surface area parameter for constituent groups 1,..., k,..., K
        q_k : array
            volume parameters for constituent groups 1,..., k,..., K
        a_inter : array
            K x K array with the interaction parameter between gropus m and n
            at the (m, n) position. Note that a_inter[k, k] = 0
        v_matrix : array
            N x K array. Each row contains the number of occurrences of the group k
            in the molecule i (i = 1,...,N)
        dmd : bool (default: False)
            if True, use the Dortmund variation of UNIFAC [1], which implies that
            b_inter and c_inter cannot be None (N x K arrays like a_inter)

        Returns
        -------

        gamma : array
            activity coefficients for the mixture(s). It has the same size as x_i

        Notes
        -----
        * This implementation is based on [2] for pure UNIFAC (dmd=False). The
          implemented thermodynamic models (pure UNIFAC and UNIFAC-Dortmund)
          are described in detail in [1] and [2]
        * If x_i is multi-dimensional with shape P x N, then temp must be a
          P-sized, 1-D array

        References
        ----------
        [1] (1) Gmehling, J.; Li, J.; Schiller, M. Ind. Eng. Chem. Res. 1993,
        32 (1), 178-193.

        [2] Fredenslund, A.; Jones, R. L.; Prausnitz, J. M. AIChE J. 1975, 21 (6),
        1086-1099.

        """

        if x_i is None:
            x_i = self.mole_frac

        if temp is None:
            temp = self.temp

        def get_gamma_log(psi_matrix, theta_vec, q_vec):
            psi_divided = psi_matrix / np.dot(psi_matrix.T, theta_vec)

            log_gamma = q_vec * (1 - np.log(np.dot(psi_matrix.T, theta_vec)) -
                                 np.dot(psi_divided, theta_vec))

            return log_gamma

        a_inter = self.a_unifac
        b_inter = self.b_unifac
        c_inter = self.c_unifac
        r_k = self.Rk
        q_k = self.Qk

        v_matrix = self.vk

        num_groups = len(a_inter)

        x_i = np.asarray(x_i)
        temp = np.asarray(temp)
        one_dimensional_output = False

        if x_i.ndim == 1:
            one_dimensional_output = True
            x_i = x_i[np.newaxis, :]
            temp = temp[np.newaxis]

        num_comp = x_i.shape[1]

        # --------------- Combinatorial term
        r_i = np.dot(v_matrix, r_k)
        q_i = np.dot(v_matrix, q_k)

        V = r_i / (r_i * x_i).sum(axis=1)[:, np.newaxis]
        V_prime = r_i**(3/4) / (x_i * r_i**(3/4)).sum(axis=1)[:, np.newaxis]
        F = q_i / (q_i * x_i).sum(axis=1)[:, np.newaxis]

        g_comb = 1 - V_prime + np.log(V_prime) - 5 * q_i * \
            (1 - V/F + np.log(V/F))

        # --------------- Residual term
        x_upper_i = v_matrix / v_matrix.sum(axis=1)[:, np.newaxis]
        theta_i = q_k * x_upper_i / np.dot(x_upper_i, q_k)[:, np.newaxis]

        g_resid = []
        for frac, tp in zip(x_i, temp):
            x_upper = np.dot(v_matrix.T, frac) / \
                np.sum(v_matrix * frac[:, np.newaxis])
            theta = q_k * x_upper / np.dot(q_k, x_upper)

            psi = np.exp(-(a_inter + b_inter * tp + c_inter * tp**2) / tp)

            ln_gamma = get_gamma_log(psi, theta, q_k)
            ln_gamma_i = np.zeros((num_comp, num_groups))

            for ind, row in enumerate(theta_i):
                ln_gamma_i[ind] = get_gamma_log(psi, row, q_k)

            g_res = v_matrix * (ln_gamma - ln_gamma_i)
            g_res = g_res.sum(axis=1)

            g_resid.append(g_res)

        g_resid = np.vstack(g_resid)
        g_total = g_comb + g_resid
        gamma = np.exp(g_total)

        if one_dimensional_output:
            return np.squeeze(gamma)
        else:
            return gamma
