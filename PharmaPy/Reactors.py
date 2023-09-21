# -*- coding: utf-8 -*-


from assimulo.solvers import CVode, LSODAR
from assimulo.problem import Explicit_Problem

from PharmaPy.Phases import classify_phases
from PharmaPy.Commons import (reorder_sens, plot_sens, trapezoidal_rule,
                              eval_state_events, handle_events,
                              unpack_states, unpack_discretized,
                              complete_dict_states, flatten_states,
                              retrieve_pde_result)

from PharmaPy.Streams import LiquidStream
from PharmaPy.Connections import get_inputs_new

from PharmaPy.Plotting import plot_function, plot_distrib
from PharmaPy.Results import DynamicResult
from PharmaPy.CheckModule import check_modeling_objects

import numpy as np
# from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

import copy
from itertools import cycle

linestyles = cycle(['-', '--', '-.', ':'])

gas_ct = 8.314  # J/mol/K
eps = np.finfo(float).eps


def check_stoichiometry(stoich, mws):
    mass_bce = np.dot(stoich, mws)

    if not np.allclose(mass_bce, 0):
        print("Warning! Material balance closure not attained with "
              "provided stoichiometric matrix. "
              "Check 'stoich_matrix' argument passed to the "
              "aggregated Kinetic instance.")


def order_state_names(names):
    material = []
    energy = []

    for name in names:
        if 'temp' in name or 'u_int' in name:
            energy.append(name)
        else:
            material.append(name)

    return material + energy


def get_sundials_callable(events, eval_sens, param_vals, unit_model, get_jac):
    flag_events = len(events) > 0

    if not flag_events and not eval_sens:
        def call_fun(t, y): return unit_model(t, y, None, None)

        def jac_fun(t, y): return get_jac(t, y, None, None, param_vals)

        kwargs_problem = {}

    # This is redundant but helps with readability
    elif flag_events and eval_sens:
        def call_fun(t, y, sw, params):
            return unit_model(t, y, sw=sw, params=params)

        def jac_fun(t, y, sw, params):
            return get_jac(t, y, sw, None, params)

        kwargs_problem = {'sw0': [True] * len(events), 'p0': param_vals}

    elif flag_events:
        def call_fun(t, y, sw):
            return unit_model(t, y, sw=sw, params=param_vals)

        def jac_fun(t, y, sw): return get_jac(t, y, sw, None, param_vals)

        kwargs_problem = {'sw0': [True] * len(events)}

    elif eval_sens:
        def call_fun(t, y, params):
            return unit_model(t, y, params=params)

        def jac_fun(t, y, params): return get_jac(t, y, None, None, param_vals)

        kwargs_problem = {'p0': param_vals}

    return call_fun, jac_fun, kwargs_problem


class _BaseReactor:
    """
    Base constructor for the reactor class.

    Parameters
    ----------
    :partic_species : list of str
        Names of the species participating in the reaction. Names
        correspond to species names in the physical properties
        .json file.
    :mask_params : list of bool (optional)
        Binary list of which parameters to exclude from the kinetics
        computations.
    :base_units : TODO: [deprecated? or unused?]
    :temp_ref : float (optional) TODO: [only active on CSTRs?]
        Reference temperature for enthalpy calculations.
    isothermal : bool
        Boolean value indicating whether the energy balance is
        considered. (i.e. dT/dt = 0 when isothermal is True)
    reset_states : bool (optional)
        Boolean value indicating whether the states should be
        reset before simulation.
    controls : dict of functions (optional)
        Dictionary with keys representing the state which is
        controlled and the value indicating the function to use
        while computing the variable. Functions are of the form
        f(time) = state_value
    return_sens : bool (optional, default = True)
        whether or not the paramest_wrapper method should return
        the sensitivity system along with the concentratio profiles.
        Use False if you want the parameter estimation platform to
        estimate the sensitivity system using finite differences
    state_events : lsit of dict(s)
        list of dictionaries, each one containing the specification of a
        state event
    """
    def __init__(self, mask_params,
                 base_units, temp_ref, isothermal,
                 reset_states, controls,
                 h_conv, ht_mode, return_sens, state_events):

        self.distributed_uo = False
        self.is_continuous = False

        self.elapsed_time = 0
        self.h_conv = h_conv
        self.area_ht = None
        self._Utility = None

        # Names
        self.bipartite = None
        self.names_upstream = None

        self.states_uo = ['mole_conc']
        self.names_states_out = ['mole_conc']
        self.states_out_dict = {}

        self.permut = None
        self.names_upstream = None

        self.ht_mode = ht_mode

        self.temp_ref = temp_ref

        self.isothermal = isothermal

        # ---------- Modeling objects
        self._Phases = None
        self._Kinetics = None
        self.mask_params = mask_params

        self.sensit = None
        self.return_sens = return_sens

        # Outputs
        self.reset_states = reset_states
        self.elapsed_time = 0

        self.reset_states = reset_states

        self.__original_prof__ = {
            'tempProf': [], 'concProf': [], 'timeProf': [],
            'volProf': [],
            'elapsed_time': 0
        }

        self.temp_control = None
        self.resid_time = None
        self.oper_mode = None
        if controls is None:
            self.controls = {}
        else:
            self.controls = controls

        if state_events is None:
            state_events = []

        self.state_event_list = state_events

        # Outputs
        self.time_runs = []
        self.temp_runs = []
        self.conc_runs = []
        self.vol_runs = []
        self.tempHt_runs = []

        self.profiles_runs = []

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases_list):
        if isinstance(phases_list, list) or isinstance(phases_list, tuple):
            self._Phases = phases_list
        elif 'LiquidPhase' in phases_list.__class__.__name__:
            self._Phases = [phases_list]
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')
        classify_phases(self)

        self.name_species = self.Liquid_1.name_species
        self.num_species = len(self.name_species)

        names = ('mole_conc', 'temp')
        dims = (self.num_species, 1)
        dict_states_out = dict(zip(names, dims))

        if self.states_out_dict.keys():
            self.states_out_dict['Liquid_1'].update(dict_states_out)
        else:
            self.states_out_dict['Liquid_1'] = dict_states_out

        self.vol_phase = copy.copy(self.Liquid_1.vol)
        self.__original_phase_dict__ = copy.deepcopy(self.Liquid_1.__dict__)
        self.__original_phase__ = copy.deepcopy(self.Liquid_1)

    @property
    def Kinetics(self):
        return self._Kinetics

    @Kinetics.setter
    def Kinetics(self, instance):
        self._Kinetics = instance
        self.partic_species = instance.partic_species

        name_params = self._Kinetics.name_params
        if self.mask_params is None:
            self.mask_params = [True] * self._Kinetics.num_params
            self.name_params = name_params

        else:
            self.name_params = [name for ind, name in enumerate(name_params)
                                if self.mask_params[ind]]

        self.mask_params = np.array(self.mask_params)

        ind_true = np.where(self.mask_params)[0]
        ind_false = np.where(~self.mask_params)[0]

        self.params_fixed = self.Kinetics.concat_params()[ind_false]

        self.ind_maskpar = np.argsort(np.concatenate((ind_true, ind_false)))

    @property
    def Utility(self):
        return self._Utility

    @Utility.setter
    def Utility(self, utility):
        self.u_ht = 1 / (1 / self.h_conv + 1 / utility.h_conv)
        self._Utility = utility

    def reset(self):
        copy_dict = copy.deepcopy(self.__original_prof__)

        self.Liquid_1.__dict__.update(self.__original_phase_dict__)
        self.__dict__.update(copy_dict)

        self.profiles_runs = []

    def _eval_state_events(self, time, states, sw):
        is_PFR = self.__class__.__name__ == 'PlugFlowReactor'

        events = eval_state_events(
            time, states, sw, self.dim_states,
            self.states_uo, self.state_event_list, sdot=self.derivatives,
            discretized_model=is_PFR)

        return events

    def heat_transfer(self, temp, temp_ht, vol):
        # Heat transfer area
        if self.ht_mode == 'coil':  # Half pipe heat transfer
            pass
        else:
            area_ht = 4 / self.diam * vol + self.area_base  # m**2
            heat_transf = self.u_ht * area_ht * (temp - temp_ht)

        return heat_transf

    def set_names(self):
        mask_species = [True] * self.num_species
        if self.name_species is not None:
            mask_species = [name in self.partic_species
                            for name in self.name_species]

        self.mask_species = np.asarray(mask_species)

        if self.__class__.__name__ == 'BatchReactor':
            index_conc = self.Kinetics.partic_species
        else:
            index_conc = self.name_species

        dim_conc = len(index_conc)

        self.states_di = {
            'mole_conc': {'index': index_conc, 'dim': dim_conc,
                          'units': 'mol/L', 'type': 'diff'}
            }

        self.fstates_di = {
            'q_rxn': {'units': 'W', 'dim': 1},
            'q_ht': {'units': 'W', 'dim': 1},
            }

        reactor_type = self.__class__.__name__

        if self.isothermal:
            self.fstates_di['temp'] = {'units': 'K', 'dim': 1, 'type': 'diff'}
        else:
            self.states_di['temp'] = {'units': 'K', 'dim': 1, 'type': 'diff'}
            if reactor_type != 'PlugFlowReactor':
                self.states_di['temp_ht'] = {'units': 'K', 'dim': 1,
                                             'type': 'diff'}

        if reactor_type == 'SemibatchReactor':
            self.states_di['vol'] = {'units': 'm**3', 'dim': 1, 'type': 'diff'}

        for key in self.states_di:
            self.states_di[key]['depends_on'] = ['time']

        if reactor_type == 'PlugFlowReactor':
            for key, val in self.states_di.items():
                self.states_di[key]['depends_on'].append('vol')

        name_states = list(self.states_di.keys())

        self.name_states = order_state_names(name_states)
        self.dim_states = [a['dim'] for a in self.states_di.values()]

        # Input names
        len_in = [self.num_species, 1, 1]
        states_in_dict = dict(zip(self.names_states_in, len_in))

        self.states_in_dict = {'Inlet': states_in_dict}

    def get_inputs(self, time):
        inlet = getattr(self, 'Inlet', None)
        if inlet is None:
            inputs = {}
        else:
            inputs = get_inputs_new(time, inlet, self.states_in_dict)

        return inputs

    def unit_model(self, time, states, sw=None, params=None):
        # Calculate inlets
        u_values = self.get_inputs(time)

        # Decompose states
        di_states = unpack_states(states, self.dim_states, self.name_states)

        di_states = complete_dict_states(time, di_states,
                                         ('vol', 'temp', 'temp_ht'),
                                         self.Liquid_1, self.controls)

        self.Liquid_1.temp = di_states['temp']

        material_bces = self.material_balances(time, **di_states,
                                               inputs=u_values)

        if 'temp' in self.states_uo:
            energy_bce = self.energy_balances(time, **di_states,
                                              inputs=u_values)

            balances = np.append(material_bces, energy_bce)
        else:
            balances = material_bces

        self.derivatives = balances

        return balances

    def get_jacobians(self, time, states, sw, sens, params, wrt_states=True):
        """
        Function that calculates df/dy (jac_states) or the rhs of the
        sensitivity system, i.e. df/dy * sens + df/dtheta
        where df/dtheta is jac_params

        Parameters
        ----------
        time : float
            integration time.
        states : array-like
            states of the ODE system.
        sw : list of bools
            list indicating the states of the switches.
        sens : array-like
            current sensitivities of the system.
        params : array-like
            value of the kinetic parameters.
        wrt_states : bool, optional
            if True, jac_states is returned (function called by problem.jac).
            Otherwise, the rhs of the sensitivity system is returned
            (function called by problem.rhs). The default is True.

        Returns
        -------
        See 'wrt_states' argument above

        """
        num_states = len(states)

        # ---------- w.r.t. states
        num_species = self.Kinetics.num_species
        conc = states[:num_species]
        if self.isothermal:
            temp = self.Liquid_1.temp
            jac_states = self.Kinetics.derivatives(conc, temp)
        else:
            temp = states[num_species]
            jac_states = np.zeros((num_states, num_states))

            jac_r_kin = self.Kinetics.derivatives(conc, temp)
            jac_states[:num_states - 1, :num_states - 1] = jac_r_kin

        if wrt_states:
            return jac_states
        else:  # ---------- w.r.t params
            jac_theta_kin = self.Kinetics.derivatives(
                conc, temp, dstates=False)

            if self.isothermal:
                jac_params = jac_theta_kin
            else:
                # jac_params = np.zeros((num_states, num_par))
                pass  # TODO: include temp row in the jacobian

            dsens_dt = jac_states.dot(sens) + jac_params

            return dsens_dt

    def flatten_states(self):
        di_out = flatten_states(self.profiles_runs)
        return di_out

    def paramest_wrapper(self, params, t_vals, modify_phase=None,
                         modify_controls=None, reord_sens=True, run_args={}):

        self.reset()

        if isinstance(modify_phase, dict):
            self.Liquid_1.updatePhase(**modify_phase)

        if isinstance(modify_controls, dict):
            self.params_control = modify_controls

        self.Kinetics.set_params(params)
        self.elapsed_time = 0

        if self.return_sens:
            t_prof, states, sens = self.solve_unit(time_grid=t_vals,
                                                   verbose=False,
                                                   eval_sens=True, **run_args)

            if reord_sens:
                sens = reorder_sens(sens)
            else:
                sens = np.stack(sens)

            c_prof = states[:, :self.Kinetics.num_species]

            return c_prof, sens

        else:
            t_prof, states = self.solve_unit(time_grid=t_vals,
                                             verbose=False,
                                             eval_sens=False, **run_args)

            c_prof = states[:, :self.Kinetics.num_species]

            return c_prof

    def plot_profiles(self, pick_comp=None, **fig_kwargs):
        """
        Plot representative profiles for tank reactors. For a more flexible
        plotting interface, see plot_function in th PharmaPy.Plotting module

        Parameters
        ----------
        pick_comp : list of str/int, optional
            list of components to be plotted. Each element of the list
            can be either the name of a species (str) or the index of the
            species (int). The default is None.
        **fig_kwargs : keyword arguments to plt.subplots()
            named arguments passed to the plotting functions. A yypical field
            is 'figsize', passed as a (width, height) tuple.

        Returns
        -------
        fig : TYPE
            fig object.
        ax : numpy array or array
            ax object or array of objects.

        """

        if pick_comp is None:
            states_plot = ('mole_conc', 'temp', 'q_rxn', 'q_ht')
        else:
            states_plot = (['mole_conc', pick_comp], 'temp', 'q_rxn', 'q_ht')

        figmap = (0, 1, 2, 2)
        ylabels = ('C_j', 'T', 'Q_rxn', 'Q_ht')

        ncols = max(figmap) + 1

        fig, ax = plot_function(self, states_plot, fig_map=figmap,
                                ncols=ncols, ylabels=ylabels, **fig_kwargs)

        if hasattr(self.result, 'temp_ht'):
            ax[1].plot(self.result.time, self.result.temp_ht, '--')

            ax[1].legend(('$T_{reactor}$', '$T_{ht}$'))

        for axis in ax:
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        fig.tight_layout()

        fig.text(0.5, 0, 'time (s)', ha='center')

        return fig, ax

    def plot_sens(self, fig_size=None, mode='per_parameter',
                  black_white=False, time_div=1):
        if self.sensit is None:
            raise AttributeError("No sensitivities detected. Run the unit "
                                 " with 'eval_sens'=True")
        self.flatten_states()

        name_states = ["C_{" + self.name_species[ind] + "}"
                       for ind in range(len(self.name_species))
                       if self.mask_species[ind]]

        if mode == 'per_parameter':
            sens_data = self.sensit
        elif mode == 'per_state':
            sens_data = reorder_sens(self.sensit, separate_sens=True)

        fig, axis = plot_sens(self.result.time, sens_data,
                              name_states=name_states,
                              name_params=self.Kinetics.name_params,
                              mode=mode, black_white=black_white,
                              time_div=time_div)

        return fig, axis


class BatchReactor(_BaseReactor):
    """Inherited constructor for the Batch reactor class.
    
    Parameters
    ---------
    partic_species : list of str
        Names of the species participating in the reaction. Names
        correspond to species names in the physical properties
        .json file.
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computations.
    base_units : str (optional, default = 'concentration')
        Basis used for material units in the reactor.
    temp_ref : float (optional, default = 298.15)
        Reference temperature for enthalpy calculations.
    isothermal : bool (optional, default = True)
        Boolean value indicating whether the energy balance is
        considered. (i.e. dT/dt = 0 when isothermal is True)
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be
        reset before simulation.
    controls : dict of functions (optional, default = None)
        Dictionary with keys representing the state which is
        controlled and the value indicating the function to use
        while computing the variable. Functions are of the form
        f(time) = state_value
    h_conv : float (optional, default = 1000)
        Convective heat transfer coefficient for the liquid phase in the reactor (W m\ :sup:`-2` K\ :sup:`-1`). 
    ht_mode : str (optional, default = 'jacket')
        What method is used for heat transfer. Options: ['jacket',
        'coil', 'bath']
    return_sens : bool (optional, default = True)
        whether or not the paramest_wrapper method should return
        the sensitivity system along with the concentratio profiles.
        Use False if you want the parameter estimation platform to
        estimate the sensitivity system using finite differences
    """

    def __init__(self, mask_params=None,
                 base_units='concentration', temp_ref=298.15,
                 isothermal=True, reset_states=False, controls=None,
                 h_conv=1000, ht_mode='jacket', return_sens=True,
                 state_events=None):

        super().__init__(mask_params,
                         base_units, temp_ref, isothermal,
                         reset_states, controls,
                         h_conv, ht_mode, return_sens, state_events)

        self.oper_mode = 'Batch'
        self.is_continuous = False
        self.nomenclature()

        self.vol_offset = 0.75

        self.material_from_upstream = False

    def nomenclature(self):
        if not self.isothermal and 'temp' not in self.controls.keys():
            self.states_uo.append('temp')

            if self.ht_mode == 'jacket':
                self.states_uo.append('temp_ht')

        self.names_states_in = self.names_states_out + ['temp', 'vol']
        self.names_states_out = self.names_states_in

    def material_balances(self, time, mole_conc, vol, temp, temp_ht, inputs):

        if self.Kinetics.keq_params is None:
            rate = self.Kinetics.get_rxn_rates(mole_conc, temp)
        else:
            concentr = np.zeros(len(self.name_species))
            concentr[self.mask_species] = mole_conc
            concentr[~self.mask_species] = self.conc_inert
            deltah_rxn = self.Liquid_1.getHeatOfRxn(
                self.Kinetics.stoich_matrix, temp, self.mask_species,
                self.Kinetics.delta_hrxn, self.Kinetics.tref_hrxn)

            rate = self.Kinetics.get_rxn_rates(mole_conc, temp,
                                               delta_hrxn=deltah_rxn)

        dmaterial_dt = rate

        return dmaterial_dt

    def energy_balances(self, time, mole_conc, vol, temp, temp_ht, inputs,
                        heat_prof=False):

        temp = np.atleast_1d(temp)
        mole_conc = np.atleast_2d(mole_conc)

        conc_all = np.ones((len(mole_conc), len(self.Liquid_1.mole_conc)))

        if mole_conc.ndim == 1:
            conc_all[self.mask_species] = mole_conc
            conc_all[~self.mask_species] *= self.conc_inert
        else:
            conc_all[:, self.mask_species] = mole_conc
            conc_all[:, ~self.mask_species] *= self.conc_inert

        # Enthalpy calculations
        _, cp_j = self.Liquid_1.getCpPure(temp)  # J/mol

        # Heat of reaction
        delta_href = self.Kinetics.delta_hrxn
        stoich = self.Kinetics.stoich_matrix
        tref_hrxn = self.Kinetics.tref_hrxn
        deltah_rxn = self.Liquid_1.getHeatOfRxn(
            stoich, temp, self.mask_species, delta_href, tref_hrxn)  # J/mol

        rates = self.Kinetics.get_rxn_rates(mole_conc, temp,
                                            overall_rates=False,
                                            delta_hrxn=deltah_rxn)

        # Balance terms (W)
        #source_term = -inner1d(deltah_rxn, rates) * vol * 1000  # vol in L
        # TODO: Check if this is correct
        # source_term = -np.dot(deltah_rxn, rates) * vol * 1000  # vol in L
        source_term = (deltah_rxn * rates).sum(axis=1) * vol * 1000  # vol in L

        if heat_prof:
            if 'temp' in self.controls.keys():
                heat_profile = -np.column_stack((source_term, ))
                capacitance = vol[0] * (conc_all *
                                        1000 * cp_j).sum(axis=1)  # J/K (NCp)
                self.capacitance = capacitance

            if self.isothermal:
                heat_profile = -np.column_stack((source_term, -source_term))
            else:
                ht_term = self.heat_transfer(temp, temp_ht, vol)
                heat_profile = np.column_stack((source_term, -ht_term))

            return heat_profile
        else:
            ht_term = self.heat_transfer(temp, temp_ht, vol)
            capacitance = vol * np.dot(conc_all * 1000, cp_j)  # J/K (NCp)
            dtemp_dt = (source_term - ht_term) / capacitance  # K/s

            if 'temp_ht' in self.states_uo:
                ht_dict = self.Utility.get_inputs(time)

                flow_ht = ht_dict['vol_flow']
                tht_in = ht_dict['temp_in']

                cp_ht = self.Utility.cp
                rho_ht = self.Utility.rho

                vol_ht = vol * 0.15  # heuristic

                dtht_dt = flow_ht / vol_ht * (tht_in - temp_ht) + \
                    ht_term / rho_ht / vol_ht / cp_ht

                output = np.array([dtemp_dt, dtht_dt])
            else:
                output = dtemp_dt

            return output

        return dtemp_dt

    def solve_unit(self, runtime=None, time_grid=None, eval_sens=False,
                   params_control=None, verbose=True, sundials_opts=None):
        """
        Batch reactor method for solving the individual unit directly.
        runtime : float (default = None)
            Value for total unit runtime.
        time_grid : list of float (optional, default = None)
            Optional list of time values for the integrator to use
            during simulation.
        eval_sens : bool (optional, default = False)
            Boolean value indicating whether the parametric
            sensitivity system will be included during simulation.
            Must be true to access sensitivity information.     
        verbose : bool (optional, default = True)
            Boolean value indicating whether the simulator will
            output run statistics after simulation is complete.
            Use true if you want to see the number of function
            evaluations and wall-clock runtime for the unit.
        timesim_limit : float (optional, default = 0)
            Float value of the maximum wall-clock time for the
            simulator to use before aborting the simulation.
        return : default 2 arrays (3 if eval_sens is True)
            Returns 2 or 3 indexed data structures. First, the
            integrator time points. Second, the state values
            corresponding to those integrator time points. And
            if eval_sens is True, third is the parametric
            sensitivity information of the simulation.
        """

        check_modeling_objects(self)

        self.set_names()

        # check_stoichiometry(self.Kinetics.stoich_matrix,
        #                     self.Liquid_1.mw[self.mask_species])

        self.params_control = params_control

        if runtime is not None:
            final_time = runtime + self.elapsed_time

        if time_grid is not None:
            final_time = time_grid[-1] + self.elapsed_time
            self.elapsed_time = time_grid[0]

        # Initial states
        conc_init = self.Liquid_1.mole_conc[self.mask_species]
        self.conc_inert = self.Liquid_1.mole_conc[~self.mask_species]
        self.num_concentr = len(conc_init)

        self.args_inputs = (self, self.num_concentr, 0)

        states_init = conc_init
        if 'temp' in self.states_uo:
            states_init = np.append(states_init, self.Liquid_1.temp)

            if 'temp_ht' in self.states_uo:
                tht_init = self.Utility.temp_in
                states_init = np.append(states_init, tht_init)

        # Create problem
        merged_params = self.Kinetics.concat_params()

        call_fn, jac_fn, kw_problem = get_sundials_callable(
            self.state_event_list, eval_sens, merged_params,
            self.unit_model, self.get_jacobians)

        problem = Explicit_Problem(call_fn, states_init, t0=self.elapsed_time,
                                   **kw_problem)

        if eval_sens:
            problem.jac = jac_fn

            def rhs_sens(t, y, sens, params): return self.get_jacobians(
                    t, y, None, sens, params, wrt_states=False)

            problem.rhs_sens = rhs_sens

        else:
            if self.isothermal and self.Kinetics.df_dstates is not None:
                problem.jac = jac_fn

        if len(self.state_event_list) > 0:
            def new_handle(solver, info):
                return handle_events(solver, info, self.state_event_list,
                                     any_event=True)

            problem.state_events = self._eval_state_events
            problem.handle_event = new_handle

        vol_tank = self.Liquid_1.vol / self.vol_offset
        self.diam = (4 / np.pi * vol_tank)**(1/3)
        self.area_base = np.pi/4 * self.diam**2

        self.derivatives = call_fn(self.elapsed_time, states_init,
                                   *list(kw_problem.values()))

        # Set solver
        solver = CVode(problem)

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        if eval_sens:
            solver.sensmethod = 'SIMULTANEOUS'
            solver.report_continuously = True

        if not verbose:
            solver.verbosity = 50

        # Solve model
        time, states = solver.simulate(final_time, ncp_list=time_grid)

        # Store results
        self.retrieve_results(time, states)

        if eval_sens:
            sensit = []
            for elem in solver.p_sol:
                sens = np.array(elem)
                sens[0] = 0  # correct NaN's at t = 0 for sensitivities
                sensit.append(sens)

            self.sensit = sensit

            return time, states, sensit
        else:
            return time, states

    def retrieve_results(self, time, states):
        time = np.asarray(time)

        # ---------- Prepare dict of results
        dp = unpack_states(states, self.dim_states, self.name_states)
        dp['time'] = time

        dp = complete_dict_states(time, dp, ('vol', 'temp'), self.Liquid_1,
                                  self.controls)

        if 'temp_ht' in self.name_states:
            heat_prof = self.energy_balances(**dp, inputs=None, heat_prof=True)
        else:
            heat_prof = self.energy_balances(temp_ht=None, **dp, inputs=None,
                                             heat_prof=True)

        dp['q_rxn'] = heat_prof[:, 0]
        dp['q_ht'] = heat_prof[:, 1]

        self.profiles_runs.append(dp)
        dp = self.flatten_states()  # In case the UO has been run before

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        # Heat duty
        self.heat_duty = np.array([trapezoidal_rule(time, dp['q_ht']), 0])  # J
        self.duty_type = [0, 0]

        # Final state
        self.elapsed_time = time[-1]

        self.Liquid_1.temp = dp['temp'][-1]

        concentr_final = self.Liquid_1.mole_conc.copy().astype(np.float64)
        concentr_final[self.mask_species] = dp['mole_conc'][-1]
        self.Liquid_1.updatePhase(vol=self.Liquid_1.vol,
                                  mole_conc=concentr_final)

        self.Outlet = self.Liquid_1
        # self.outputs = states
        self.outputs = dp


class CSTR(_BaseReactor):
    """
    Inherited constructor for the continuous stirred-tank reactor (CSTR) class.
    
    Parameters
    ----------
    partic_species : list of str
        Names of the species participating in the reaction. Names
        correspond to species names in the physical properties
        .json file.
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computations.
    base_units : str (optional, default = 'concentration')
        Basis used for material units in the reactor.
    temp_ref : float (optional, default = 298.15)
        Reference temperature for enthalpy calculations.
    isothermal : bool (optional, default = True)
        Boolean value indicating whether the energy balance is
        considered. (i.e. dT/dt = 0 when isothermal is True)
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be
        reset before simulation.
    controls : dict of functions (optional, default = None)
        Dictionary with keys representing the state which is
        controlled and the value indicating the function to use
        while computing the variable. Functions are of the form
        f(time) = state_value
    h_conv : float (optional, default = 1000)
        Convective heat transfer coefficient for the liquid phase in the reactor (W m\ :sup:`-2` K\ :sup:`-1`). 
    ht_mode : str (optional, default = 'jacket')
        What method is used for heat transfer. Options: ['jacket',
        'coil', 'bath']
    return_sens : bool (optional, default = True)
        whether or not the paramest_wrapper method should return
        the sensitivity system along with the concentratio profiles.
        Use False if you want the parameter estimation platform to
        estimate the sensitivity system using finite differences
    """

    def __init__(self, mask_params=None,
                 base_units='concentration', temp_ref=298.15,
                 isothermal=True, reset_states=False, controls=None,
                 h_conv=1000, ht_mode='jacket', return_sens=True,
                 state_events=None):
        
        super().__init__(mask_params,
                         base_units, temp_ref, isothermal,
                         reset_states, controls,
                         h_conv, ht_mode, return_sens, state_events)

        self._Inlet = None
        self.oper_mode = 'Continuous'
        self.is_continuous = True
        self.nomenclature()

        self.vol_offset = 0.75

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet_object):
        self._Inlet = inlet_object

    def nomenclature(self):
        # self.name_species = self.Liquid_1.name_species

        if not self.isothermal:
            self.states_uo.append('temp')
            if self.ht_mode == 'jacket':
                self.states_uo.append('temp_ht')

        self.names_states_out += ['temp', 'vol_flow']
        self.names_states_in = self.names_states_out

    def material_balances(self, time, mole_conc, vol, temp, temp_ht, inputs):
        inputs = inputs['Inlet']
        inlet_flow = inputs['vol_flow']
        inlet_conc = inputs['mole_conc']

        if self.Kinetics.keq_params is None:
            rate = self.Kinetics.get_rxn_rates(mole_conc[self.mask_species],
                                               temp)
        else:
            deltah_rxn = self.Liquid_1.getHeatOfRxn(temp,
                                                    self.Kinetics.tref_hrxn)

            rate = self.Kinetics.get_rxn_rates(mole_conc[self.mask_species],
                                               temp,
                                               deltah_rxn)

        rates = np.zeros_like(mole_conc)
        rates[self.mask_species] = rate

        dmaterial_dt = inlet_flow / vol * (inlet_conc - mole_conc) + rates

        return dmaterial_dt

    def energy_balances(self, time, mole_conc, vol, temp, temp_ht, inputs,
                        heat_prof=False):

        inputs = inputs['Inlet']

        inlet_flow = inputs['vol_flow']
        inlet_conc = inputs['mole_conc']
        inlet_temp = inputs['temp']

        temp = np.atleast_1d(temp)

        # Enthalpy calculations
        _, cp_j = self.Liquid_1.getCpPure(temp)  # J/mol
        h_tempj = self.Liquid_1.getEnthalpy(temp, self.temp_ref, total_h=False,
                                            basis='mole')

        # Heat of reaction
        deltah_ref = self.Kinetics.delta_hrxn
        tref_dh = self.Kinetics.tref_hrxn

        deltah_rxn = self.Liquid_1.getHeatOfRxn(
            self.Kinetics.stoich_matrix, temp, self.mask_species,
            deltah_ref, tref_dh)  # J/mol

        rates = self.Kinetics.get_rxn_rates(mole_conc.T[self.mask_species].T,
                                            temp, overall_rates=False,
                                            delta_hrxn=deltah_rxn)

        # Inlet stream
        stream = self.Inlet
        h_inj = stream.getEnthalpy(inlet_temp, temp_ref=self.temp_ref,
                                   total_h=False, basis='mole')

        h_in = (inlet_conc * h_inj).sum(axis=1) * 1000  # J/m**3
        h_temp = (mole_conc * h_tempj).sum(axis=1) * 1000  # J/m**3
        flow_term = inlet_flow * (h_in - h_temp)  # W

        # Balance terms (W) - convert vol to L
        # source_term = -inner1d(deltah_rxn, rates) * vol * 1000
        # TODO: Check if this is correct
        # source_term = -np.dot(deltah_rxn, rates) * vol * 1000  # vol in L
        source_term = (deltah_rxn * rates).sum(axis=1) * vol * 1000  # vol in L

        if heat_prof:
            if self.isothermal:
                ht_term = -(source_term + flow_term)
            else:
                ht_term = self.heat_transfer(temp, temp_ht, vol)

            heat_profile = np.column_stack((source_term, -ht_term,
                                            flow_term))
            return heat_profile
        else:
            ht_term = self.heat_transfer(temp, temp_ht, vol)

            div = vol * np.dot(mole_conc * 1000, cp_j)  # J/K (NCp)
            dtemp_dt = (flow_term + source_term - ht_term) / div  # K/s

            if 'temp_ht' in self.states_uo:
                ht_controls = self.Utility.get_inputs(time)
                tht_in = ht_controls['temp_in']
                flow_ht = ht_controls['vol_flow']

                cp_ht = self.Utility.cp
                rho_ht = self.Utility.rho

                if 'vol' in self.states_uo:  # Semibatch
                    vol_ht = self.vol_ht
                else:
                    vol_ht = vol * 0.15

                dtht_dt = flow_ht / vol_ht * (tht_in - temp_ht) + \
                    ht_term / rho_ht / vol_ht / cp_ht

                output = np.array([dtemp_dt, dtht_dt])
            else:
                output = dtemp_dt

            return output

    def solve_unit(self, runtime=None, time_grid=None, eval_sens=False,
                   params_control=None, verbose=True, sundials_opts=None):

        check_modeling_objects(self)

        self.params_control = params_control
        self.set_names()

        self.num_concentr = len(self.Liquid_1.mole_conc)
        self.args_inputs = (self, self.num_concentr, 0)

        if runtime is not None:
            final_time = runtime + self.elapsed_time

        if time_grid is not None:
            final_time = time_grid[-1] + self.elapsed_time

        # Reset states
        if self.reset_states:
            self.reset()

        vol_tank = self.Liquid_1.vol / self.vol_offset
        self.diam = (4 / np.pi * vol_tank)**(1/3)
        self.area_base = np.pi/4 * self.diam**2

        # # Define inlet streams
        # self.Inlet.Liquid_1.getProps()

        # Initial states
        states_init = self.Liquid_1.mole_conc

        if 'temp' in self.states_uo:
            states_init = np.append(states_init, self.Liquid_1.temp)
            if 'temp_ht' in self.states_uo:
                tht_init = self.Utility.evaluate_inputs(0)['temp_in']
                states_init = np.append(states_init, tht_init)

        self.resid_time = self.Liquid_1.vol / self.Inlet.vol_flow

        # Create problem
        merged_params = self.Kinetics.concat_params()
        if eval_sens:
            pass
        else:
            def fobj(time, states): return self.unit_model(
                time, states, merged_params)

            problem = Explicit_Problem(fobj, states_init,
                                       t0=self.elapsed_time)

        # Set solver
        solver = CVode(problem)
        # solver = LSODAR(problem)

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        if not verbose:
            solver.verbosity = 50

        # Solve model
        time, states = solver.simulate(final_time, ncp_list=time_grid)

        # Store results
        self.statesProf = states
        self.states = states[-1]

        self.retrieve_results(time, states)
        # self.flatten_states()

        return time, states

    def retrieve_results(self, time, states):
        time = np.asarray(time)

        # ---------- Prepare dict of results
        dp = unpack_states(states, self.dim_states, self.name_states)
        dp['time'] = time

        dp = complete_dict_states(time, dp, ('vol', 'temp'), self.Liquid_1,
                                  self.controls)

        inputs = self.get_inputs(time)

        if 'temp_ht' in self.name_states:
            heat_prof = self.energy_balances(**dp, inputs=inputs,
                                             heat_prof=True)
        else:
            heat_prof = self.energy_balances(temp_ht=None, **dp, inputs=inputs,
                                             heat_prof=True)

        dp['q_rxn'] = heat_prof[:, 0]
        dp['q_ht'] = heat_prof[:, 1]

        self.profiles_runs.append(dp)

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        # Outlet stream/phase
        path = self.Inlet.path_data
        if self.__class__.__name__ == 'SemibatchReactor':
            self.Liquid_1.updatePhase(temp=dp['temp'][-1],
                                      mole_conc=dp['mole_conc'][-1],
                                      vol=dp['vol'][-1])
            self.Outlet = self.Liquid_1
        else:
            self.Outlet = LiquidStream(path, temp=dp['temp'][-1],
                                       mole_conc=dp['mole_conc'][-1],
                                       vol_flow=inputs['Inlet']['vol_flow'][-1])

        # Output vector
        outputs = {key: dp[key] for key in ('mole_conc', 'temp')}
        outputs['vol_flow'] = inputs['Inlet']['vol_flow']
        self.outputs = outputs


class SemibatchReactor(CSTR):
    """
    Inherited constructor for the semibatch stirred-tank reactor class. This method inherits from the CSTR constructor.
    
    Parameters
    ----------
    partic_species : list of str
        Names of the species participating in the reaction. Names
        correspond to species names in the physical properties
        .json file.
    vol_tank : float
        Volume of the vessel in m**3. Required to ensure that the
        vessel does not overflow
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computations.
    base_units : str (optional, default = 'concentration')
        Basis used for material units in the reactor.
    temp_ref : float (optional, default = 298.15)
        Reference temperature for enthalpy calculations.
    isothermal : bool (optional, default = True)
        Boolean value indicating whether the energy balance is
        considered. (i.e. dT/dt = 0 when isothermal is True)
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be
        reset before simulation.
    controls : dict of functions (optional, default = None)
        Dictionary with keys representing the state which is
        controlled and the value indicating the function to use
        while computing the variable. Functions are of the form
        f(time) = state_value
    h_conv : float (optional, default = 1000)
        Convective heat transfer coefficient for the liquid phase in the reactor (W m\ :sup:`-2` K\ :sup:`-1`). 
    ht_mode : str (optional, default = 'jacket')
        What method is used for heat transfer. Options: ['jacket',
        'coil', 'bath']
    return_sens : bool (optional, default = True)
        whether or not the paramest_wrapper method should return
        the sensitivity system along with the concentratio profiles.
        Use False if you want the parameter estimation platform to
        estimate the sensitivity system using finite differences
    """
    
    def __init__(self, vol_tank,
                 mask_params=None,
                 base_units='concentration', temp_ref=298.15,
                 isothermal=True, reset_states=False, controls=None,
                 h_conv=1000, ht_mode='jacket', return_sens=True,
                 state_events=None):
        

        super().__init__(mask_params,
                         base_units, temp_ref,
                         isothermal, reset_states, controls,
                         h_conv, ht_mode, return_sens, state_events)

        self.oper_mode = 'Semibatch'
        self.is_continuous = False

        self.diam = (4/np.pi * vol_tank)**(1/3)  # m
        self.vol_ht = vol_tank * 0.15
        self.area_base = np.pi/4 * self.diam**2

        self.material_from_upstream = False

    def nomenclature(self):
        self.states_uo.append('vol')
        if not self.isothermal:
            self.states_uo.append('temp')

            if self.ht_mode == 'jacket':
                self.states_uo.append('temp_ht')

        self.names_states_in = self.names_states_out + ['temp', 'vol_flow']
        self.names_states_out += ['temp', 'vol']

    def material_balances(self, time, inputs, **kwargs):
        dc_dt = super().material_balances(time, inputs=inputs, **kwargs)

        dvol_dt = inputs['Inlet']['vol_flow']

        return np.append(dc_dt, dvol_dt)

    def solve_unit(self, runtime=None, time_grid=None, eval_sens=False,
                   params_control=None, verbose=True, sundials_opts=None):


        check_modeling_objects(self)

        self.params_control = params_control
        self.set_names()

        if runtime is not None:
            final_time = runtime + self.elapsed_time

        if time_grid is not None:
            final_time = time_grid[-1] + self.elapsed_time

        # Reset states
        if self.reset_states:
            self.reset()

        # Initial states
        states_init = self.Liquid_1.mole_conc
        states_init = np.append(states_init, self.Liquid_1.vol)

        if 'temp' in self.states_uo:
            states_init = np.append(states_init, self.Liquid_1.temp)
            if 'temp_ht' in self.states_uo:
                tht_init = self.Utility.evaluate_inputs(0)['temp_in']
                states_init = np.append(states_init, tht_init)

        merged_params = self.Kinetics.concat_params()
        if eval_sens:
            pass
        else:
            def fobj(time, states): return self.unit_model(
                time, states, merged_params)

            problem = Explicit_Problem(fobj, states_init,
                                       t0=self.elapsed_time)

        # Set solver
        solver = CVode(problem)

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        solver.sensmethod = 'SIMULTANEOUS'
        solver.report_continuously = True

        if not verbose:
            solver.verbosity = 50

        # Solve model
        time, states = solver.simulate(final_time, ncp_list=time_grid)

        # Store results
        self.time_runs.append(time)
        self.statesProf = states
        self.states = states[-1]

        self.retrieve_results(time, states)
        self.flatten_states()

        return time, states

    # def retrieve_results(self, time, states):
    #     conc_prof = states[:, :self.num_concentr]
    #     vol_prof = states[:, self.num_concentr]

    #     if self.isothermal:
    #         temp_prof = np.ones_like(time) * self.Liquid_1.temp
    #         tht_prof = None

    #     elif self.temp_control is not None:
    #         conc_prof = states.copy()
    #         temp_prof = self.temp_control(**self.params_control['temp'])

    #     else:
    #         temp_prof = states[:, self.num_concentr + 1]

    #         if 'temp_ht' in self.states_uo:
    #             tht_prof = states[:, -1]
    #         else:
    #             tht_prof = None

    #     # Heat profile
    #     u_inputs = self.get_inputs(time)

    #     self.heat_prof = self.energy_balances(time, conc_prof, vol_prof,
    #                                           temp_prof,
    #                                           tht_prof, u_inputs,
    #                                           heat_prof=True)

    #     self.temp_runs.append(temp_prof)
    #     self.conc_runs.append(conc_prof)
    #     self.vol_runs.append(vol_prof)

    #     if tht_prof is not None:
    #         self.tempHt_runs.append(tht_prof)

    #     # Final state
    #     self.elapsed_time = time[-1]
    #     self.concentr = self.conc_runs[-1][-1]
    #     self.temp = self.temp_runs[-1][-1]
    #     self.vol = self.vol_runs[-1][-1]

    #     self.Liquid_1.temp = self.temp
    #     self.Liquid_1.vol = self.vol
    #     self.Liquid_1.updatePhase(vol=self.vol, mole_conc=self.concentr)
    #     self.Outlet = self.Liquid_1
    #     self.outputs = states


class PlugFlowReactor(_BaseReactor):
    """
    Inherited constructor for the plug-flow (PFR) reactor class.
    
    Parameters
    ----------
    diam_in : float
        Diameter of the tubular reactor in meters.
    num_discr : int
        number of finite volumes to use to discretize the volume coordinate
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computations.
    base_units : str (optional, default = 'concentration')
        Basis used for material units in the reactor.
    temp_ref : float (optional, default = 298.15)
        Reference temperature for enthalpy calculations.
    isothermal : bool (optional, default = True)
        Boolean value indicating whether the energy balance is
        considered. (i.e. dT/dt = 0 when isothermal is True)
    adiabatic : bool (optional, default = False)
        Boolean value indication whether the reactor is considered
        to be adiabatic. Temperature change will not be zero but
        heat transfer will be zero in this case.
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be
        reset before simulation.
    controls : dict of functions (optional, default = None)
        Dictionary with keys representing the state which is
        controlled and the value indicating the function to use
        while computing the variable. Functions are of the form
        f(time) = state_value
    h_conv : float (optional, default = 1000)
        Convective heat transfer coefficient for the liquid phase in the reactor (W m\ :sup:`-2` K\ :sup:`-1`). 
    ht_mode : str (optional, default = 'bath')
        What method is used for heat transfer. Options: ['jacket',
        'coil', 'bath']
    return_sens : bool (optional, default = True)
        whether or not the paramest_wrapper method should return
        the sensitivity system along with the concentratio profiles.
        Use False if you want the parameter estimation platform to
        estimate the sensitivity system using finite differences
    """
    
    def __init__(self, diam_in, num_discr,
                 mask_params=None,
                 base_units='concentration', temp_ref=298.15,
                 isothermal=False, adiabatic=False,
                 reset_states=False, controls=None,
                 h_conv=1000, ht_mode='bath', return_sens=True,
                 state_events=None):
        

        super().__init__(mask_params,
                         base_units, temp_ref, isothermal,
                         reset_states, controls,
                         h_conv, ht_mode, return_sens, state_events)

        self.is_continuous = True
        self.oper_mode = 'Continuous'
        self.diam = diam_in
        self.vol_offset = 1

        self.num_discr = num_discr

        self.adiabatic = adiabatic

        # Distributed system attributes
        self.xPositions = []
        self.distributed_uo = True

        self.nomenclature()

        self._Inlet = None
        self.tau = None

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet_object):
        self._Inlet = inlet_object

        if self.states_out_dict.keys():
            self.states_out_dict['Liquid_1']['vol_flow'] = 1
        else:
            self.states_out_dict['Liquid_1'] = {'vol_flow': 1}

        num_comp = len(self.Inlet.name_species)
        len_in = [num_comp, 1, 1]
        states_in_dict = dict(zip(self.names_states_in, len_in))

        self.states_in_dict = {'Inlet': states_in_dict}

    def nomenclature(self):
        if not self.isothermal:
            self.states_uo.append('temp')

        self.names_states_out += ['temp', 'vol_flow']
        self.names_states_in = self.names_states_out

    def get_inputs(self, time):
        inputs = get_inputs_new(time, self.Inlet, self.states_in_dict)

        return inputs

    def material_steady(self, conc, temp):
        if self.Kinetics.keq_params is None:
            rate = self.Kinetics.get_rxn_rates(conc, temp)
        else:
            deltah_rxn = self.Liquid_1.getHeatOfRxn(
                self.Kinetics.stoich_matrix,
                temp,
                self.mask_species,
                self.Kinetics.delta_hrxn,
                self.Kinetics.tref_hrxn)

            rate = self.Kinetics.get_rxn_rates(conc, temp,
                                               delta_hrxn=deltah_rxn)

        dconc_dv = rate / self.Inlet.vol_flow

        return dconc_dv

    def energy_steady(self, conc, temp):
        _, cp_j = self.Liquid_1.getCpPure(temp)

        concentr = np.zeros_like(self.Liquid_1.mole_conc)
        concentr[self.mask_species] = conc
        concentr[~self.mask_species] = self.c_inert

        # Volumetric heat capacity
        cp_vol = np.dot(cp_j, concentr) * 1000  # W/K

        # Heat of reaction
        delta_href = self.Kinetics.delta_hrxn
        stoich = self.Kinetics.stoich_matrix
        tref_hrxn = self.Kinetics.tref_hrxn

        deltah_rxn = self.Liquid_1.getHeatOfRxn(
            stoich, temp, self.mask_species, delta_href, tref_hrxn)  # J/mol

        rates = self.Kinetics.get_rxn_rates(conc, temp, overall_rates=False,
                                            delta_hrxn=deltah_rxn)

        # ---------- Balance terms (W)
        # source_term = -inner1d(deltah_rxn, rates) * 1000  # W/m**3
        # TODO: Check if this is correct
        # source_term = -np.dot(deltah_rxn, rates) * 1000  # W / m**3
        source_term = (deltah_rxn * rates).sum(axis=1) * 1000  # W / m**3

        if self.adiabatic:
            heat_transfer = 0
        else:  # W/m**3
            a_prime = self.diam / 4  # m**2 / m**3
            heat_transfer = self.u_ht * a_prime * (temp - self.Utility.temp)

        flow_term = self.Inlet.vol_flow * cp_vol

        # -------- Energy balance
        dtemp_dv = (source_term - heat_transfer) / flow_term

        return dtemp_dv

    def unit_steady(self, time, states, params=None):
        conc = states[:self.num_species]

        if 'temp' in self.states_uo:
            temp = states[self.num_species]
        else:
            temp = self.Inlet.temp

        material_bce = self.material_steady(conc, temp)

        if 'temp' in self.states_uo:
            energy_bce = self.energy_steady(conc, temp)

            deriv = np.append(material_bce, energy_bce)
        else:
            deriv = material_bce

        return deriv

    def solve_steady(self, vol_rxn, adiabatic=False):
        self.adiabatic = adiabatic
        self.set_names()

        if adiabatic:
            self.isothermal = False
            self.states_uo.append('temp')

        c_inlet = self.Inlet.concentr

        self.c_inert = c_inlet[~self.mask_species]
        c_partic = c_inlet[self.mask_species]

        self.num_species = len(c_partic)

        states_init = c_partic

        if 'temp' in self.states_uo:
            states_init = np.append(states_init, self.Inlet.temp)

        problem = Explicit_Problem(self.unit_steady, states_init, t0=0)
        solver = CVode(problem)

        volPosition, states_solver = solver.simulate(vol_rxn)

        num_x = len(volPosition)

        # Retrieve results
        concentr = states_solver[:, :self.num_species]
        self.concProfSteady = concentr
        self.volPosition = volPosition
        if 'temp' in self.states_uo:
            temper = states_solver[:, self.num_species]
        else:
            temper = np.ones(num_x) * self.Inlet.temp

        self.tempProfSteady = temper

        return volPosition, states_solver

    def material_balances(self, time, mole_conc, vol_diff, temp, flow_in,
                          rate_j):
        # Inputs

        # Finite differences
        diff_conc = np.diff(mole_conc, axis=0)

        rates = np.zeros((len(mole_conc), self.num_species))
        rates[:, self.mask_species] = rate_j

        dconc_dt = -flow_in*(diff_conc.T / vol_diff).T + \
            rates[1:]  # TODO Is it correct to ignore rate at V = 0?

        return dconc_dt

    def energy_balances(self, time, mole_conc, vol_diff, temp, flow_in, rate_i,
                        heat_profile=False):

        _, cp_j = self.Liquid_1.getCpPure(temp)

        # Volumetric heat capacity
        # cp_vol = inner1d(cp_j, mole_conc) * 1000  # J/m**3/K
        # TODO: Check if this is correct
        # cp_vol = -np.dot(cp_j, mole_conc) * 1000  # J/m**3/K
        cp_vol = (cp_j * mole_conc).sum(axis=1) * 1000  # vol in L


        # Heat of reaction
        delta_href = self.Kinetics.delta_hrxn
        stoich = self.Kinetics.stoich_matrix
        tref_hrxn = self.Kinetics.tref_hrxn

        deltah_rxn = self.Liquid_1.getHeatOfRxn(
            stoich, temp, self.mask_species, delta_href, tref_hrxn)  # J/mol

        # ---------- Balance terms (W)
        # source_term = -inner1d(deltah_rxn, rate_i * 1000)  # W/m**3
        # TODO: Check if this is correct
        # source_term = -np.dot(deltah_rxn, rate_i * 1000)  # W/m**3
        source_term = (deltah_rxn * rate_i * 1000).sum(axis=1)  # W/m**3

        temp_diff = np.diff(temp)
        flow_term = -flow_in * temp_diff / vol_diff  # K/s

        if self.adiabatic:
            heat_transfer = np.zeros_like(self.vol_discr)

        elif self.isothermal:
            heat_transfer = source_term
        else:  # W/m**3
            a_prime = 4 / self.diam  # m**2 / m**3

            temp_ht = self.Utility.get_inputs(time)['temp_in']
            heat_transfer = self.u_ht * a_prime * (temp - temp_ht)  # W/m**3

        if heat_profile:
            ht_total = trapezoidal_rule(self.result.vol, heat_transfer)  # W
            return ht_total

        else:
            dtemp_dt = flow_term + \
                (source_term[1:] - heat_transfer[1:])/cp_vol[1:]

            return dtemp_dt  # TODO: if adiabatic, T vs V shouldn't be constant

    def unit_model(self, time, states, sw=None, params=None, enrgy_bce=False):

        di_states = unpack_discretized(states, self.len_states,
                                       self.name_states)

        inputs = self.get_inputs(time)['Inlet']

        di_states = complete_dict_states(
            time, di_states, ('mole_conc', 'temp'), self.Liquid_1,
            self.controls, inputs, num_discr=self.num_discr)

        flow_in = inputs['vol_flow']  # m**3/s

        # Include left boundary
        temp_all = di_states['temp']

        vol_diff = np.diff(self.vol_discr)

        # Reaction rates
        conc_partic = di_states['mole_conc'][:, self.mask_species]
        if self.Kinetics.keq_params is None:
            rates_i = self.Kinetics.get_rxn_rates(
                conc_partic, temp_all, overall_rates=False)
        else:
            deltah_rxn = self.Liquid_1.getHeatOfRxn(
                self.Kinetics.stoich_matrix,
                temp_all,
                self.mask_species,
                self.Kinetics.delta_hrxn,
                self.Kinetics.tref_hrxn)

            rates_i = self.Kinetics.get_rxn_rates(conc_partic,
                                                  temp_all,
                                                  delta_hrxn=deltah_rxn,
                                                  overall_rates=False)

        rates_j = np.dot(rates_i, self.Kinetics.normalized_stoich.T)

        material_bces = self.material_balances(time, **di_states,
                                               vol_diff=vol_diff,
                                               flow_in=flow_in, rate_j=rates_j)
        if enrgy_bce:
            ht_inst = self.energy_balances(time, **di_states,
                                           vol_diff=vol_diff,
                                           flow_in=flow_in, rate_i=rates_i,
                                           heat_profile=True)
            return ht_inst

        elif 'temp' in self.states_uo:
            energy_bce = self.energy_balances(time, **di_states,
                                              vol_diff=vol_diff,
                                              flow_in=flow_in, rate_i=rates_i,
                                              heat_profile=False)

            balances = np.column_stack((material_bces, energy_bce)).ravel()
        else:
            balances = material_bces.ravel()

        self.derivatives = balances

        return balances

    def _get_tau(self):
        vol_rxn = self.Liquid_1.vol

        time_upstream = getattr(self.Inlet, 'time_upstream')
        if time_upstream is None:
            time_upstream = [0]

        inputs = self.get_inputs(time_upstream[-1])['Inlet']
        tau = vol_rxn / inputs['vol_flow']

        self.tau = tau

        return tau

    def solve_unit(self, runtime=None, time_grid=None, verbose=True,
                   any_event=True, sundials_opts=None):


        check_modeling_objects(self)

        if runtime is not None:
            final_time = runtime + self.elapsed_time

        if time_grid is not None:
            final_time = time_grid[-1] + self.elapsed_time

        self.set_names()

        vol_rxn = self.Liquid_1.vol
        self.vol_discr = np.linspace(0, vol_rxn, self.num_discr + 1)

        c_init = np.ones((self.num_discr, self.num_species)) * \
            self.Liquid_1.mole_conc

        self.num_states = self.num_species

        # c_init = c_init.astype(np.float64)
        # c_init[c_init == 0] = eps

        self.num_concentr = self.num_species  # TODO: make consistent with Batch
        self.args_inputs = (self, self.num_concentr, 0)

        len_states = [self.num_species]

        if 'temp' in self.states_uo:
            temp_init = np.ones(len(c_init)) * self.Liquid_1.temp
            states_init = np.column_stack((c_init, temp_init)).ravel()
            self.num_states += 1
            len_states.append(1)
        else:
            states_init = c_init.ravel()

        self.trim_idx = np.cumsum(len_states)[:-1]
        self.len_states = len_states

        model = self.unit_model

        model, jac_fn, kw_model = get_sundials_callable(
            self.state_event_list, eval_sens=False, param_vals=[],
            unit_model=self.unit_model, get_jac=self.get_jacobians)

        problem = Explicit_Problem(model, states_init, t0=self.elapsed_time,
                                   **kw_model)

        if len(self.state_event_list) > 0:
            # def model(t, y): return self.unit_model(t, y, None)
            # problem = Explicit_Problem(model, states_init,
            #                            t0=self.elapsed_time)
        # else:
            # switches = [True] * len(self.state_event_list)
            # problem = Explicit_Problem(self.unit_model, states_init,
            #                            t0=self.elapsed_time, sw0=switches)

            def new_handle(solver, info):
                return handle_events(solver, info, self.state_event_list,
                                     any_event=any_event)

            problem.state_events = self._eval_state_events
            problem.handle_event = new_handle

        self.derivatives = model(self.elapsed_time, states_init,
                                 *list(kw_model.values()))

        solver = CVode(problem)
        solver.linear_solver = 'SPGMR'

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        if not verbose:
            solver.verbosity = 50

        time, states_solver = solver.simulate(final_time, ncp_list=time_grid)

        self.retrieve_results(time, states_solver)

        return time, states_solver

    def retrieve_results(self, time, states):
        time = np.asarray(time)

        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        inputs = self.get_inputs(time)['Inlet']

        dp = unpack_discretized(states, self.dim_states, self.name_states,
                                indexes=indexes, inputs=inputs)

        dp = complete_dict_states(time, dp, ('temp', ),
                                  self.Liquid_1, self.controls,
                                  num_discr=self.num_discr + 1)  # + inputs

        dp['time'] = time
        dp['vol'] = self.vol_discr

        self.profiles_runs.append(dp)

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        outlet_states = retrieve_pde_result(self.result, x_name='vol',
                                            x=self.vol_discr[-1])

        outlet_states['mole_conc'] = np.column_stack(
            list(outlet_states['mole_conc'].values()))

        # dm_dt = 0
        # mole_frac = self.Liquid_1.conc_to_frac(outlet_states['mole_conc'],
        #                                        basis='mole')
        # dens = self.Liquid_1.getDensity(mole_frac=mole_frac, basis='mole')

        # mole_frac_in = self.Liquid_1.conc_to_frac(inputs['mole_conc'],
        #                                           basis='mole')
        # dens_in = self.Liquid_1.getDensity(mole_frac=mole_frac_in,
        #                                    basis='mole')

        outlet_states['vol_flow'] = np.ones_like(time) * inputs['vol_flow']

        self.outputs = outlet_states

        # Outlet stream
        path = self.Inlet.path_data
        self.Outlet = LiquidStream(path, temp=outlet_states['temp'][-1],
                                   mole_conc=outlet_states['mole_conc'][-1],
                                   vol_flow=outlet_states['vol_flow'][-1])

        # Energy balance
        ht_time = np.zeros_like(time)
        for ind, row in enumerate(states):
            ht_time[ind] = -self.unit_model(time[ind], row, enrgy_bce=True)

        self.heat_profile = ht_time
        self.heat_duty = np.array([trapezoidal_rule(time, ht_time), 0])
        self.duty_type = [0, 0]

    # def flatten_states(self):
    #     if type(self.timeProf) is list:
    #         self.concProf = np.vstack(self.concProf)

    def plot_steady(self, fig_size=None, title=None):
        fig, axes = plt.subplots(1, 2, figsize=fig_size)

        # Concentration
        axes[0].plot(self.volPosition, self.concProfSteady)
        axes[0].set_ylabel('$C_j$ (mol/L)')
        axes[0].legend(self.name_species)

        axes[1].plot(self.volPosition, self.tempProfSteady)
        axes[1].set_ylabel('$T$ (K)')

        for ax in axes:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        fig.text(0.5, 0, 'V ($m^3$)')
        fig.tight_layout()

        fig.suptitle(title)

        return fig, axes

    def plot_profiles(self, times=None, vol=None, pick_comp=None, **fig_kw):

        if pick_comp is None:
            pick_comp = np.arange(self.num_species)
        else:
            pick_comp = pick_comp

        states_plot = [('mole_conc', pick_comp), 'temp']

        y_labels = ('C_j', 'T')

        fig, ax = plot_distrib(self, states_plot, times=times, x_vals=vol,
                               x_name='vol', ncols=2,
                               ylabels=y_labels, **fig_kw)

        if times is not None:

            fig.tight_layout()

        elif vol is not None:  # TODO: use new Plotting.py module
            if vol is None:
                raise RuntimeError('Please provide a volume value using '
                                   "the argument 'vol'")

            fig.text(0.5, 0, '$t$ ($s$)', ha='center')

        for axis in ax:
            axis.grid()

            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)

            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        if len(ax) == 1:
            ax = ax[0]

        fig.tight_layout()

        return fig, ax

    def animate_reactor(self, filename=None, step_data=2, fps=5,
                        pick_idx=None, title=None):

        if filename is None:
            filename = 'anim'

        if pick_idx is None:
            pick_idx = np.arange(self.num_species)
            names = self.name_species
        else:
            pick_idx = pick_idx
            names = [self.name_species[ind] for ind in pick_idx]

        fig_anim, (ax_anim, ax_temp) = plt.subplots(2, 1, figsize=(4, 5))
        ax_anim.set_xlim(0, self.vol_discr.max())
        fig_anim.suptitle(title)

        fig_anim.subplots_adjust(left=0, bottom=0, right=1, top=1,
                                 wspace=None, hspace=None)

        conc_min = np.vstack(self.concPerVolElem)[:, pick_idx].min()
        conc_max = np.vstack(self.concPerVolElem)[:, pick_idx].max()

        conc_diff = conc_max - conc_min

        ax_anim.set_ylim(conc_min - 0.03*conc_diff, conc_max + conc_diff*0.03)

        ax_anim.set_xlabel(r'$V$ ($m^3$)')
        ax_anim.set_ylabel('$C_j$ (mol/L)')

        temp = self.tempProf
        temp_diff = temp.max() - temp.min()
        ax_temp.set_xlim(0, self.vol_discr.max())
        ax_temp.set_ylim(temp.min() - temp_diff*0.03,
                         temp.max() + temp_diff*0.03)

        ax_temp.set_xlabel(r'$V$ ($m^3$)')
        ax_temp.set_ylabel('$T$ (K)')

        def func_data(ind):
            conc_species = []
            for comp in pick_idx:
                conc_species.append(self.concPerSpecies[comp][ind])

            conc_species = np.column_stack(conc_species)
            return conc_species

        lines_conc = ax_anim.plot(self.vol_discr, func_data(0))
        line_temp, = ax_temp.plot(self.vol_discr, temp[0])

        time_tag = ax_anim.text(
            1, 1.04, '$time = {:.1f}$ s'.format(self.timeProf[0]),
            horizontalalignment='right',
            transform=ax_anim.transAxes)

        def func_anim(ind):
            f_vals = func_data(ind)
            for comp, line in enumerate(lines_conc):
                line.set_ydata(f_vals[:, comp])
                line.set_label(names[comp])

            line_temp.set_ydata(temp[ind])

            ax_anim.legend()
            fig_anim.tight_layout()

            time_tag.set_text('$t = {:.1f}$ s'.format(self.timeProf[ind]))

        frames = np.arange(0, len(self.timeProf), step_data)
        animation = FuncAnimation(fig_anim, func_anim, frames=frames,
                                  repeat=True)

        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'),
                              bitrate=-1)

        suff = '.mp4'

        animation.save(filename + suff, writer=writer)

        return animation, fig_anim, (ax_anim, ax_temp)
