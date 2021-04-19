#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:56:33 2019

@author: casas100
"""


import numpy as np
# from autograd import numpy as np

gas_ct = 8.314  # J/mol/K
eps = np.finfo(float).eps


class RxnKinetics:

    def __init__(self, stoich_matrix, k_params, ea_params,
                 keq_params=None, params_f=None,
                 delta_hrxn=None, tref_hrxn=None,
                 temp_ref=298.15, reparam_center=True,
                 kinetic_model=None, df_dstates=None, df_dtheta=None):
        """ Create a reactor object

        Parameters
        ----------
        stoiciometric_matrix : numpy array
            stoichiometric matrix for the set of reactions. It must have
            n_rxn rows and n_comp columns, so the element (i, j) represents
            the coefficient of species j in reaction i
        kin_model : callable (optional)
            kinetic model to be used to compute reaction rates. It must have
            the signature:

                >>> kin_model(conc, params, *args)

            where `conc` is the concentrations of the participating species,
            `params` are the kinetic parameters (as a list or tuple)
        params_k : dict of tuples
            parameters for the temperature-dependent term in the kinetic
            model. It must be a dictionary with the following structure:
                {'k_vals': (phi_1, phi_2, ...), 'E_vals': (Ea_1, Ea_2, ...)}
            The keys must be as shown
        rxn_orders : list
            parameters for the concentration-dependent term in the kinetic
            model. If not given, the reaction orders are set to the stoichio-
            metric coefficients


        """

        self.temp_ref = temp_ref

        self.args_kin = ()

        # ---------- Kinetic model
        self.elem_flag = False
        if kinetic_model is None:
            self.kinetic_model = self.elem_f_model
            self.df_dstates = self.elem_df_dstates
            self.df_dthetaf = self.elem_df_dtheta

            self.elem_flag = True
        else:
            self.kinetic_model = kinetic_model
            self.df_dstates = df_dstates
            self.df_dthetaf = df_dtheta

        # ---------- Parameters
        self.reparam_center = reparam_center

        self.k_params = np.atleast_1d(k_params)
        self.ea_params = np.atleast_1d(ea_params)


        # Note: self.rxn_orders also includes the 'orders' for products
        # self.num_paramsk = len(self.k_params)
        self.num_paramsk = len(self.k_params) + len(self.ea_params)

        if params_f is None:
            # self.params_concat = params_k.T.flatten()
            self.rxn_orders = None

        elif self.elem_flag:
            params_f = np.asarray(params_f)
            if params_f.ndim == 1:
                params_f = params_f[np.newaxis, ...]

            self.rxn_orders = params_f

        self.params_f = params_f
        self.order_map = stoich_matrix < 0

        # Reparametrizations
        self.phi_1 = None
        self.phi_2 = None

        if keq_params is None:
            self.keq_params = keq_params
        else:
            self.keq_params = np.atleast_1d(keq_params)

        # Heat of reaction
        if delta_hrxn is None:
            self.delta_hrxn = delta_hrxn
            self.tref_hrxn = tref_hrxn
        else:
            self.delta_hrxn = np.atleast_1d(delta_hrxn)
            if tref_hrxn is None:
                self.tref_hrxn = temp_ref
            else:
                self.tref_hrxn = tref_hrxn

        self.set_stoichiometry(stoich_matrix)

        # Outputs
        self.rxn_rates = None
        self.time_profile = None
        self.conc_profile = None
        self.sensitivities = None

    def set_stoichiometry(self, stoich_matrix):

        stoich_matrix = np.atleast_2d(stoich_matrix)
        self.num_rxns, self.num_species = stoich_matrix.shape

        # Transform parameters
        kvals = np.atleast_1d(self.k_params)
        evals = np.atleast_1d(self.ea_params)

        ea_term = evals/gas_ct/self.temp_ref * self.reparam_center
        self.phi_1 = np.log(kvals) - ea_term
        self.phi_2 = np.log(evals/gas_ct)

        # Normalize stoichiometric coefficients
        first_negative = (stoich_matrix < 0).argmax(axis=1)
        ref_stoich = np.zeros(self.num_rxns)

        for ind in range(self.num_rxns):
            ref_stoich[ind] = stoich_matrix[ind, first_negative[ind]]

        self.normalized_stoich = stoich_matrix.T / abs(ref_stoich)
        self.stoich_matrix = stoich_matrix

        # Names
        num_kpar = len(kvals)
        name_k = ['\\phi_{1, %i}' % ind for ind in range(1, num_kpar + 1)]
        name_e = ['\\phi_{2, %i}' % ind for ind in range(1, num_kpar + 1)]

        if self.params_f is None:
            name_orders = []
        else:
            num_orders = (stoich_matrix < 0).sum()
            name_orders = [r'\alpha_{}'.format(ind)
                           for ind in range(1, num_orders + 1) ]

        self.name_params = name_k + name_e + name_orders
        self.num_params = len(self.name_params)

    def concat_params(self):

        params_k_conc = np.concatenate((self.phi_1, self.phi_2))
        if self.elem_flag:
            if self.params_f is None:  # rxn orders are fixed
                params_concat = params_k_conc
            else:
                orders = self.params_f[self.order_map]
                params_concat = np.concatenate((params_k_conc, orders))

        else:
            params_concat = np.concatenate((params_k_conc, self.params_f))

        return params_concat

    def assign_params(self, params):  # From 1D params to matrix-shaped params

        self.phi_1 = params[:self.num_rxns]
        self.phi_2 = params[self.num_rxns:self.num_rxns*2]

        if self.elem_flag:
            # Reorganize rxn orders into a stoich_matrix-like structure
            if self.params_f is not None:
                self.rxn_orders = np.zeros_like(self.stoich_matrix,
                                                dtype=np.float64)

                self.rxn_orders[self.order_map] = params[self.num_paramsk:]

    def temp_term(self, temp):

        temp = np.asarray(temp)
        inv_temp = (1/self.temp_ref - 1/temp)

        if temp.ndim == 0:
            k_temp = np.exp(self.phi_1 + np.exp(self.phi_2) * inv_temp)
        else:
            k_temp = np.exp(self.phi_1 +
                            np.outer(inv_temp, np.exp(self.phi_2)))

        return k_temp

    def equil_term(self, temp, deltah_temp):
        temp = np.asarray(temp)
        inv_temp = (1/temp - 1/self.tref_hrxn)

        if temp.ndim == 0:
            k_eq = self.keq_params * np.exp(-deltah_temp/gas_ct * inv_temp)
        else:
            k_eq = self.keq_params * np.exp(
                -np.outer(inv_temp, deltah_temp/gas_ct))

        return k_eq

    def dk_dkparams(self, temp):

        temp_term = self.temp_term(temp)

        drate_dphi1 = np.diag(temp_term)
        drate_dphi2 = np.diag((1/self.temp_ref - 1/temp) * temp_term)
        drate_dphi2 *= np.exp(self.phi_2)

        drate_dk = np.hstack((drate_dphi1, drate_dphi2))

        return np.atleast_2d(drate_dk)

    def elem_f_model(self, conc):
        """ Compute elementary reaction rates for each participating reaction

        Parameters
        ----------
        conc : array-like
            molar concentrations for each participating species (size n_comp)

        Returns
        -------
        rxn_rates : array
            rate for each reaction taking place in the system (size n_rxns)
        rates_species : array
            rate for each species in each reaction (size n_rxns x n_comp)
        total_rates : array
            net reaction rate for each component among all the reactions it
            participates in (size n_comp)
        """

        # Identify reactants by the sign of stoichiometic coeff
        is_reactant = self.stoich_matrix < 0
        conc = np.asarray(conc)
        n_conc = len(conc)

        if self.rxn_orders is None:
            orders = abs(is_reactant * self.stoich_matrix)
            self.rxn_orders = orders
        else:
            orders = self.rxn_orders

        # Compute elementary reaction rate
        conc = np.abs(conc)
        if conc.ndim == 1:
            f_term = np.zeros(self.num_rxns)
            for ind in range(self.num_rxns):
                f_term[ind] = np.prod(conc**(orders[ind]))
        else:  # vectorized
            f_term = np.zeros((n_conc, self.num_rxns))
            for ind in range(self.num_rxns):
                f_term[:, ind] = np.prod(conc**(orders[ind]), axis=1)

        return f_term

    def equilibrium_model(self, conc, temp, deltah_rxn):
        is_product = self.stoich_matrix > 0
        orders = abs(is_product * self.stoich_matrix)
        conc = np.asarray(conc)
        n_conc = len(conc)

        keq_temp = self.equil_term(temp, deltah_rxn)

        keq_temp[keq_temp == 0] = 1e20

        # Forward term
        f_term = self.elem_f_model(conc)

        # Backward term
        if conc.ndim == 1:
            r_term = np.zeros(self.num_rxns)
            for ind in range(self.num_rxns):
                r_term[ind] = np.prod(conc**(orders[ind])) / keq_temp[ind]
        else:
            r_term = np.zeros((n_conc, self.num_rxns))
            for ind in range(self.num_rxns):
                r_term[:, ind] = np.prod(conc**(orders[ind]), axis=1) / keq_temp[ind]

        overall_rate = f_term - r_term

        return overall_rate

    def elem_df_dstates(self, conc):

        f_term = self.elem_f_model(conc)

        df_dconc = f_term[..., np.newaxis] * self.rxn_orders / (conc + eps)

        return df_dconc

    def elem_df_dtheta(self, conc):

        f_term = self.elem_f_model(conc)

        conc_correc = np.maximum(np.ones_like(conc) * 1e-20, conc)

        num_orders = self.order_map.sum()
        drate_dorder = np.zeros((self.num_rxns, num_orders))

        for ind, row in enumerate(self.order_map):
            conc_m = conc_correc[row]  # see Section 3.2.2
            drate_dorder[ind] = np.log(conc_m) * f_term[ind]

        drate_dorder = drate_dorder

        return drate_dorder

    def derivatives(self, conc, temp, dstates=True):
        temp_terms = self.temp_term(temp)
        f_terms = self.kinetic_model(conc, *self.args_kin)

        if dstates:  # --------------- wrt states
            df_dstates = self.df_dstates(conc, *self.args_kin)
            dr_dstates = (temp_terms * df_dstates.T).T
            jac_states = np.dot(self.normalized_stoich, dr_dstates)
            return jac_states
        else:  # --------------- wrt parameters
            dk_dphi = self.dk_dkparams(temp).T
            dr_dthetak = (dk_dphi * f_terms).T

            if self.params_f is None:
                dr_dparams = dr_dthetak
            else:
                dr_dthetaf = self.df_dthetaf(
                        conc, *self.args_kin) * temp_terms

                dr_dparams = np.hstack((dr_dthetak, dr_dthetaf))

            jac_params = np.dot(self.normalized_stoich, dr_dparams)

            if jac_params.ndim == 1:
                jac_params = jac_params[..., np.newaxis]

            return jac_params

    def get_rxn_rates(self, conc, temp=298.15, overall_rates=True, jac=False,
                      delta_hrxn=None):

        if jac:
            jac_states = self.derivatives(conc, temp)
            return jac_states

        else:
            temp_terms = self.temp_term(temp)

            if self.keq_params is None:
                f_terms = self.kinetic_model(conc, *self.args_kin)
            else:
                f_terms = self.equilibrium_model(conc, temp, delta_hrxn)

            rxn_rates = temp_terms * f_terms
            if overall_rates:  # per species
                total_rates = np.dot(rxn_rates, self.normalized_stoich.T)
                return total_rates
            else:  # per rxn
                return rxn_rates


class CrystKinetics:
    """ Specify a kinetics crystallization kinetics object

        Parameters
        ----------
        coeff_solub : array-like
            coefficients for a temperature-dependent solubility (S) polynomial
            of the form S = A + B*T + C*T^2...
        b_params : array-like (3 elements)
            primary nucleation coefficients, with the result given in
            number of particles per second
        s_params : array-like (4 elements) (optional)
            secondary nucleation coefficients, with the result given in
            number of particles per second
        g_params : array-like (dimension 3) (optional)
            nucleation parameters, with the result given in um/s
        d_params : array_like (dimension 3) (optional)
            dissolution parameters, with the result given in um/s
        rel_super : bool (optional, default True)
            if True, relative supersaturation is used for computing the kinetic
            mechanism
        alpha_fn : callable (optional)
            function that receives the vector of liquid phase compositions
            and calculates the growth inhibition term, returning a float
            between 0 and 1

        Returns
        -------

    """

    def __init__(self, coeff_solub,
                 b_params=None, s_params=None, g_params=None, d_params=None,
                 rel_super=True, alpha_fn=None, temp_ref=298.15):

        self.temp_ref = temp_ref
        self.rel_super = rel_super

        # ---------- Parameters
        self.names_mechanisms = ('nucl_prim', 'nucl_sec', 'growth',
                                 'dissolution')

        params_all = [b_params, s_params, g_params, d_params]

        params_list = [item for item in params_all if item is not None]
        self.params_keys = [name for param, name
                            in zip(params_all, self.names_mechanisms)
                            if param is not None]

        self.num_par_tuple = [len(item) for item in params_list]

        self.set_params(params_list)  # create self.params

        self.coeff_solub = np.asarray(coeff_solub)

        self.name_params = ('k_{bp}', 'E_{bp}', 'b',
                            'k_{bs}', 'E_{bs}', 's_1', 's_2',
                            'k_{g}', 'E_{g}', 'g',
                            'k_{d}', 'E_{d}', 'd')

        self.num_params = len(self.name_params)

        # ---------- Growth decreasing fn
        if alpha_fn is None:
            self.alpha_fn = lambda conc: 1
        else:
            self.alpha_fn = alpha_fn

    def set_params(self, params_in):
        if all(isinstance(item, tuple) for item in params_in):
            params_dict = dict(zip(self.params_keys, params_in))
            params_complete = self.parse_params(params_dict)
        else:
            split_idx = np.array([3, 4, 3, 3]).cumsum()[:-1]
            params_in = np.split(params_in, split_idx)

            params_complete = dict(zip(self.names_mechanisms, params_in))

        self.params = params_complete

    def parse_params(self, param_dict, reparam=True):
        if reparam:
            zero_log = np.log(eps)

            nucl_prim = [zero_log, 0, 0]
            nucl_sec = [zero_log, 0, 0, 0]
            growth = [zero_log, 0, 0]
            dissol = [zero_log, 0, 0]

            params_parsed = {'nucl_prim': nucl_prim, 'nucl_sec': nucl_sec,
                             'growth': growth, 'dissolution': dissol}

            for name in self.names_mechanisms:
                if name in self.params_keys:
                    vals = param_dict[name]

                    tref = self.temp_ref
                    phi_1 = np.log(vals[0] + eps) - vals[1]/gas_ct/tref
                    phi_2 = vals[1]/gas_ct

                    params_parsed[name] = list(vals)

                    params_parsed[name][0] = phi_1
                    params_parsed[name][1] = phi_2
        else:
            nucl_prim = [0, 0, 0]
            nucl_sec = [0, 0, 0, 0]
            growth = [0, 0, 0]
            dissol = [0, 0, 0]

            params_parsed = {'nucl_prim': nucl_prim, 'nucl_sec': nucl_sec,
                             'growth': growth, 'dissolution': dissol}

            for name in self.names_mechanisms:
                if name in self.params_keys:
                    vals = param_dict[name]
                    tref = self.temp_ref
                    phi_1 = vals[0]
                    phi_2 = vals[1]

                    params_parsed[name] = list(vals)

                    params_parsed[name][0] = phi_1
                    params_parsed[name][1] = phi_2

        return params_parsed

    def concat_params(self):
        params = [np.array(vals) for vals in self.params.values()]
        params_conc = np.concatenate(params)
        return params_conc

    def get_solubility(self, temp):
        int_coeff = np.arange(len(self.coeff_solub))

        temp = np.asarray(temp)
        if temp.ndim == 0:
            c_satur = (temp**int_coeff * self.coeff_solub).sum()
        else:
            temp = temp[..., np.newaxis]
            c_satur = (temp**int_coeff * self.coeff_solub).sum(axis=1)

        return c_satur

    def get_kinetics(self, conc_target, temp, kv_cry,
                     mom_3=None, nucl_sec_out=False):

        # Supersaturation
        conc_sat = self.get_solubility(temp)
        sup_sat = (conc_target - conc_sat)

        if self.rel_super:
            sup_sat = sup_sat / conc_sat

        temp_abs = temp
        tref = self.temp_ref

        par_p, par_s, par_g, par_d = self.params.values()

        phi_p = par_p[0] + par_p[1]*(1/tref - 1/temp_abs)
        phi_s = par_s[0] + par_s[1]*(1/tref - 1/temp_abs)
        phi_g = par_g[0] + par_g[1]*(1/tref - 1/temp_abs)

        phi_d = par_d[0] + par_d[1]*(1/tref - 1/temp_abs)

        absup = np.abs(sup_sat)

        if isinstance(sup_sat, float):

            if sup_sat >= 0:
                absup = max(eps, absup)
                nucl_prim = np.exp(phi_p) * sup_sat * absup**(par_p[2] - 1)

                growth = np.exp(phi_g) * sup_sat * absup**(par_g[2] - 1)

                # growth = growth * self.alpha_fn(conc)

                if mom_3 is not None:
                    # mom_3 = mom_3 + eps
                    nucl_sec = np.exp(phi_s) * sup_sat * absup**(par_s[2] - 1) * \
                        kv_cry**par_s[3] * max(0, mom_3)**par_s[3]
                else:
                    nucl_sec = 0

                dissol = 0
            else:
                growth = 0
                nucl_prim = 0
                nucl_sec = 0

                dissol = np.exp(phi_d) * sup_sat * absup**(par_d[2] - 1)

            # Returns
            self.prim_nucl = nucl_prim
            self.sec_nucl = nucl_sec
            self.growth = growth  # um/s
            self.dissol = dissol  # um/s

        else:
            positive_map = sup_sat > 0
            sup_positive = absup[positive_map]
            sup_negative = absup[~positive_map]

            growth = np.zeros_like(sup_sat)
            growth[positive_map] = np.exp(phi_g[positive_map]) * \
                sup_positive**par_g[2]

            nucl_prim = np.zeros_like(sup_sat)
            nucl_prim[positive_map] = np.exp(phi_p[positive_map]) * \
                sup_positive**par_p[2]

            nucl_sec = np.zeros_like(sup_sat)
            if mom_3 is not None:
                mom_3 = mom_3[positive_map] + eps
                nucl_sec = np.exp(phi_s[positive_map]) * sup_positive**par_s[2] * \
                    kv_cry**par_s[3] * max(0, mom_3)**par_s[3]

            dissol = np.zeros_like(sup_sat)
            dissol[~positive_map] = np.exp(phi_d[~positive_map]) * \
                sup_negative**par_d[2]

        if nucl_sec_out:
            return nucl_prim, nucl_sec, growth, dissol
        else:
            nucl = nucl_prim + nucl_sec
            return nucl, growth, dissol

    def deriv_cryst(self, conc, temp):
        conc_sat = self.get_solubility(temp)
        ssat = max(eps, (conc - conc_sat) / conc_sat)

        def dmech_dparam(mech):
            dmech = np.array([mech,
                              mech * (1 / self.temp_ref - 1 / temp),
                              mech * np.log(ssat)])

            return dmech

        dbp_dpar = dmech_dparam(self.prim_nucl)
        dbs_dpar = dmech_dparam(self.sec_nucl)
        dgr_dpar = dmech_dparam(self.growth)
        ddiss_dpar = dmech_dparam(self.dissol)

        return dbp_dpar, dbs_dpar, dgr_dpar, ddiss_dpar, conc_sat
