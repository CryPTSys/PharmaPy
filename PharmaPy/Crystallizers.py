# -*- coding: utf-8 -*-


# import autograd.numpy as np


from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

from PharmaPy.Phases import classify_phases
from PharmaPy.Streams import LiquidStream, SolidStream
from PharmaPy.MixedPhases import Slurry, SlurryStream
from PharmaPy.Commons import (reorder_sens, plot_sens, trapezoidal_rule,
                              upwind_fvm, high_resolution_fvm,
                              eval_state_events, handle_events,
                              unpack_states, complete_dict_states,
                              flatten_states)

from PharmaPy.ProcessControl import analyze_controls

from PharmaPy.jac_module import numerical_jac, numerical_jac_central, dx_jac_x
from PharmaPy.Connections import get_inputs, get_inputs_new

from PharmaPy.Results import DynamicResult
from PharmaPy.Plotting import plot_function, plot_distrib

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LightSource

from scipy.optimize import newton

import copy
import string

# try:
#     from jax import jacfwd
#     import jax.numpy as jnp
#     # from autograd import jacobian as autojac
#     # from autograd import make_jvp
# except:
#     print()
#     print(
#         'JAX is not available to perform automatic differentiation. '
#         'Install JAX if supported by your operating system (Linux, Mac).')
#     print()

import numpy as np

eps = np.finfo(float).eps
gas_ct = 8.314  # J/mol/K


class _BaseCryst:
    """ Construct a Crystallizer Object

    Parameters
    ----------
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computations
    method : str
        Choice of the numerical method. Options are: 'moments', '1D-FVM'
    target_comp : str, list of strings
        Name of the crystallizing compound(s) from .json file.
    scale : float
        Scaling factor by which crystal size distribution will be
        multiplied.
    vol_tank : TODO - Remove, it comes from Phases module.
    controls : dict of dicts(funcs) (optional, default = None)
        Dictionary with keys representing the state(e.g.'Temp') which is
        controlled and the value indicating the function to use
        while computing the variable. Functions are of the form
        f(time) = state_value
    adiabatic : bool (optional, default=True)
        Boolean value indicating whether the heat transfer of
        the crystallization is considered.
    rad_zero : float (optional, default=TODO)
        TODO Size of the first bin of the CSD discretization [m]
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be reset
        before simulation
    h_conv : float (optional, default = TODO) (maybe remove?)
        TODO
    vol_ht : float (optional, default = TODO)
        TODO Volume of the cooling jacket [m^3]
    basis : str (optional, default = T0DO)
        TODO Options :'massfrac', 'massconc'
    jac_type : str
        TODO Options: 'AD'
    state_events : lsit of dict(s)
        list of dictionaries, each one containing the specification of a
        state event
    param_wrapper : callable (optional, default = None)
        function with the signature

            param_wrapper(states, sens)

        Useful when the parameter estimation problem is a function of the
        states y -h(y)- rather than y itself.

        'states' is a DynamicResult object and 'sens' is a dictionary
        that contains N_y number of sensitivity arrays, representing
        time-depending sensitivities. Each array in sens has dimensions
        num_times x num_params. 'param_wrapper' has to return two outputs,
        one array containing h(y) and list of arrays containing
        sens(h(y))
    """
    np = np
    # @decor_states
    
    def __init__(self, mask_params,
                 method, target_comp, scale, vol_tank, controls,
                 adiabatic, rad_zero,
                 reset_states,
                 h_conv, vol_ht, basis, jac_type,
                 state_events, param_wrapper):

        if jac_type == 'AD':
            try:
                import jax.numpy as np
                _BaseCryst.np = np
            except:
                pass

        self.distributed_uo = False
        self.mask_params = mask_params
        self.basis = basis
        self.adiabatic = adiabatic

        # ---------- Building objects
        self._Phases = None
        self._Kinetics = None
        self._Utility = None
        self.material_from_upstream = False

        self.jac_type = jac_type

        if isinstance(target_comp, str):
            target_comp = [target_comp]

        self.target_comp = target_comp

        self.scale = scale
        self.scale_flag = True
        self.vol_tank = vol_tank

        # Controls
        if controls is None:
            self.controls = {}
        else:
            self.controls = analyze_controls(controls)

        self.method = method
        self.rad = rad_zero

        self.dx = None
        self.sensit = None

        # ---------- Create jacobians (autodiff)
        self.jac_states_vals = None
        # if method == 'moments':
        #     self.jac_states_fun = autojac(self.unit_model, 1)
        #     self.jac_params_fun = autojac(self.unit_model, 2)
        # elif method == 'fvm':
        #     self.jac_states_fun = make_jvp(self.fvm_method)

        #     # self.jac_params_fun = autojac(self.fvm_method, 1)
        #     self.jac_params_fun = None

        # Outlets
        self.reset_states = reset_states
        self.elapsed_time = 0

        self.profiles_runs = []

        self.__original_prof__ = {
            'tempProf': [], 'concProf': [], 'distribProf': [], 'timeProf': [],
            'elapsed_time': 0, 'scale_flag': True
        }

        # ---------- Names
        self.states_uo = ['mass_conc']
        self.names_states_in = ['mass_conc']

        # if not self.isothermal and 'temp' not in self.controls.keys():
        #     self.states_uo.append('temp')

        self.names_upstream = None
        self.bipartite = None

        # Other parameters
        self.h_conv = h_conv

        # Slurry phase
        self.Slurry = None

        # Parameters for optimization
        self.params_iter = None
        self.vol_mult = 1

        if state_events is None:
            state_events = []

        self.state_event_list = state_events

        self.param_wrapper = param_wrapper

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if isinstance(phases, (list, tuple)):
            self._Phases = phases
        elif isinstance(phases, Slurry):
            self._Phases = phases.Phases
        elif phases.__module__ == 'PharmaPy.Phases':
            if self._Phases is None:
                self._Phases = [phases]
            else:
                self._Phases.append(phases)
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')

        if isinstance(phases, Slurry):
            self.Slurry = phases
        elif isinstance(self._Phases, (list, tuple)):
            if len(self._Phases) > 1:

                # Mixed phase
                self.Slurry = Slurry()
                self.Slurry.Phases = self._Phases

        if self.Slurry is not None:
            self.__original_phase_dict__ = [
                copy.deepcopy(phase.__dict__) for phase in self.Slurry.Phases]

            self.vol_slurry = copy.copy(self.Slurry.vol)
            if isinstance(self.vol_slurry, np.ndarray):
                self.vol_phase = self.vol_slurry[0]
            else:
                self.vol_phase = self.vol_slurry

            classify_phases(self)  # Solid_1, Liquid_1...

            # Names and target compounds
            self.name_species = self.Liquid_1.name_species

            # Input defaults
            self.input_defaults = {
                'distrib': np.zeros_like(self.Solid_1.distrib)}

            name_bool = [name in self.target_comp for name in self.name_species]
            self.target_ind = np.where(name_bool)[0][0]

            # Save safe copy of original phases
            self.__original_phase__ = [copy.deepcopy(self.Liquid_1),
                                       copy.deepcopy(self.Solid_1)]

            self.__original_phase__ = copy.deepcopy(self.Slurry)

            self.kron_jtg = np.zeros_like(self.Liquid_1.mass_frac)
            self.kron_jtg[self.target_ind] = 1

            # ---------- Names
            # Moments
            if self.method == 'moments':
                name_mom = [r'\mu_{}'.format(ind) for ind
                            in range(self.Solid_1.num_mom)]
                name_mom.append('C')

                self.num_distr = len(self.Solid_1.moments)

            else:
                self.num_distr = len(self.Solid_1.distrib)

            # Species
            if self.name_species is None:
                num_sp = len(self.Liquid_1.mass_frac)
                self.name_species = list(string.ascii_uppercase[:num_sp])

            self.states_in_dict = {
                'Liquid_1': {'mass_conc': len(self.Liquid_1.name_species)},
                'Inlet': {'vol_flow': 1, 'temp': 1}}

            self.nomenclature()

    @property
    def Kinetics(self):
        return self._Kinetics

    @Kinetics.setter
    def Kinetics(self, instance):
        self._Kinetics = instance

        name_params = self._Kinetics.name_params
        if self.mask_params is None:
            self.mask_params = [True] * self._Kinetics.num_params
            self.name_params = name_params

        else:
            self.name_params = [name for ind, name in enumerate(name_params)
                                if self.mask_params[ind]]

        self.mask_params = np.array(self.mask_params)

    @property
    def Utility(self):
        return self._Utility

    @Utility.setter
    def Utility(self, utility):
        self.u_ht = 1 / (1 / self.h_conv + 1 / utility.h_conv)
        self._Utility = utility

    def nomenclature(self):
        name_class = self.__class__.__name__

        states_di = {
            }

        di_distr = {'dim': self.num_distr,
                    'index': list(range(self.num_distr)), 'type': 'diff',
                    'depends_on': ['time', 'x_cryst']}

        if name_class != 'BatchCryst':
            self.names_states_in += ['vol_flow', 'temp']

            if self.method == 'moments':
                self.states_in_dict['Inlet']['mu_n'] = self.num_distr
            else:
                self.states_in_dict['Inlet']['distrib'] = self.num_distr

        if self.method == 'moments':
            # mom_names = ['mu_%s0' % ind for ind in range(self.num_mom)]

            # for mom in mom_names[::-1]:
            self.names_states_in.insert(0, 'mu_n')

            # self.states_in_dict['solid']['moments']

            if name_class == 'MSMPR':
                self.states_uo.append('moments')
                # self.states_in_dict['Inlet']['distrib'] = self.num_distr

                di_distr['units'] = 'm**n/m**3'
                states_di['mu_n'] = di_distr
            else:
                self.states_uo.append('total_moments')

                di_distr['units'] = 'm**n'
                states_di['mu_n'] = di_distr

                # if name_class == 'SemibatchCryst':
                    # self.states_in_dict['Inlet']['distrib'] = self.num_distr

        elif self.method == '1D-FVM':
            self.names_states_in.insert(0, 'distrib')

            states_di['distrib'] = di_distr

            if name_class == 'MSMPR':
                self.states_uo.insert(0, 'distrib')
                di_distr['units'] = '#/m**3/um'

            else:
                self.states_uo.insert(0, 'total_distrib')
                di_distr['units'] = '#/um'

        states_di['mass_conc'] = {'dim': len(self.name_species),
                                  'index': self.name_species,
                                  'units': 'kg/m**3', 'type': 'diff',
                                  'depends_on': ['time']}

        if name_class != 'MSMPR':
            states_di['vol'] = {'dim': 1, 'units': 'm**3', 'type': 'diff',
                                'depends_on': ['time']}
            self.states_uo.append('vol')

        if self.adiabatic:
            self.states_uo.append('temp')

            states_di['temp'] = {'dim': 1, 'units': 'K', 'type': 'diff',
                                 'depends_on': ['time']}
        elif 'temp' not in self.controls:
            self.states_uo += ['temp', 'temp_ht']

            states_di['temp'] = {'dim': 1, 'units': 'K', 'type': 'diff',
                                 'depends_on': ['time']}
            states_di['temp_ht'] = {'dim': 1, 'units': 'K', 'type': 'diff',
                                    'depends_on': ['time']}

        self.states_in_phaseid = {'mass_conc': 'Liquid_1'}
        self.names_states_out = self.names_states_in

        self.states_di = states_di
        self.dim_states = [di['dim'] for di in self.states_di.values()]
        self.name_states = list(self.states_di.keys())

        self.fstates_di = {
            'supersat': {'dim': 1, 'units': 'kg/m**3'},
            'solubility': {'dim': 1, 'units': 'kg/m**3'}
            }

        if 'temp' in self.controls:
            self.fstates_di['temp'] = {'dim': 1, 'units': 'K'}

        if self.method != 'moments':
            self.fstates_di['mu_n'] = {'dim': 4, 'index': list(range(4)),
                                       'units': 'm**n'}

            self.fstates_di['vol_distrib'] = {
                'dim': self.num_distr,
                'index': list(range(self.num_distr)),
                'units': 'm**3/m**3'}

    def reset(self):
        copy_dict = copy.deepcopy(self.__original_prof__)
        self.__dict__.update(copy_dict)

        for phase, di in zip(self.Phases, self.__original_phase_dict__):
            phase.__dict__.update(di)

        self.profiles_runs = []

    def get_inputs(self, time):

        if self.__class__.__name__ == 'BatchCryst':
            inputs = {}
        else:
            inputs = get_inputs_new(time, self.Inlet, self.states_in_dict)

        return inputs

    def method_of_moments(self, mu, conc, temp, params, rho_cry, vol=1):
        kv = self.Solid_1.kv

        # Kinetics
        if self.basis == 'mass_frac':
            rho_liq = self.Liquid_1.getDensity()
            comp_kin = conc / rho_liq
        else:
            comp_kin = conc

        # Kinetic terms
        mu_susp = mu*(1e-6)**np.arange(self.num_distr) / vol  # m**n/m**3_susp
        nucl, growth, dissol = self.Kinetics.get_kinetics(comp_kin, temp, kv,
                                                          mu_susp)

        growth = growth * self.Kinetics.alpha_fn(conc)

        ind_mom = np.arange(1, len(mu))

        # Model
        dmu_zero_dt = np.atleast_1d(nucl * vol)
        dmu_1on_dt = ind_mom * (growth + dissol) * mu[:-1] + \
            nucl * self.rad**ind_mom
        dmu_dt = np.concatenate((dmu_zero_dt, dmu_1on_dt))

        # Material balance in kg_API/s --> G in um, u_2 in um**2 (or m**2/m**3)
        mass_transf = np.atleast_1d(rho_cry * kv * (
            3*(growth + dissol)*mu[2] + nucl*self.rad**3)) * (1e-6)**3

        return dmu_dt, mass_transf

    def fvm_method(self, csd, moms, conc, temp, params, rho_cry,
                   output='dstates', vol=1):

        mu_2 = moms[2]

        kv_cry = self.Solid_1.kv

        # Kinetic terms
        if self.basis == 'mass_frac':
            rho_liq = self.Liquid_1.getDensity()
            comp_kin = conc / rho_liq
        else:
            comp_kin = conc

        nucl, growth, dissol = self.Kinetics.get_kinetics(comp_kin, temp,
                                                          kv_cry, moms)

        nucl = nucl * self.scale * vol

        impurity_factor = self.Kinetics.alpha_fn(conc)
        growth = growth * impurity_factor  # um/s

        dissol = dissol  # um/s

        boundary_cond = nucl / (growth + eps) # num/um or num/um/m**3
        f_aug = np.concatenate(([boundary_cond]*2, csd, [csd[-1]]))

        # Flux source terms
        f_diff = np.diff(f_aug)
        # f_diff[f_diff == 0] = eps  # avoid division by zero for theta

        if growth > 0:
            theta = f_diff[:-1] / (f_diff[1:] + eps*10)
        else:
            theta = f_diff[1:] / (f_diff[:-1] + eps*10)

        # Van-Leer limiter
        limiter = np.zeros_like(f_diff)
        limiter[:-1] = (np.abs(theta) + theta) / (1 + np.abs(theta))

        growth_term = growth * (f_aug[1:-1] + 0.5 * f_diff[1:] * limiter[:-1])
        dissol_term = dissol * (f_aug[2:] - 0.5 * f_diff[1:] * limiter[1:])

        flux = growth_term + dissol_term

        if output == 'flux':
            return flux  # TODO: isn't it necessary to divide by dx?
        elif 'dstates':
            dcsd_dt = -np.diff(flux) / self.dx

            # Material bce in kg_API/s --> G in um, mu_2 in m**2 (or m**2/m**3)
            mass_transfer = rho_cry * kv_cry * (
                3*(growth + dissol)*mu_2 + nucl*self.rad**3) * (1e-6)

            return dcsd_dt, np.array(mass_transfer)

    def unit_model(self, time, states, params=None, sw=None,
                   mat_bce=False, enrgy_bce=False):

        di_states = unpack_states(states, self.dim_states, self.name_states)

        # Inputs
        u_input = self.get_inputs(time)

        di_states = complete_dict_states(time, di_states,
                                         ('temp', 'temp_ht', 'vol'),
                                         self.Slurry, self.controls)

        # ---------- Physical properties
        self.Liquid_1.updatePhase(mass_conc=di_states['mass_conc'])
        self.Liquid_1.temp = di_states['temp']
        self.Solid_1.temp = di_states['temp']

        rhos_susp = self.Slurry.getDensity(temp=di_states['temp'])

        name_unit = self.__class__.__name__

        if self.method == 'moments':
            di_states['distrib'] = di_states['mu_n']
            moms = di_states['mu_n'] * \
                (1e-6)**np.arange(self.states_di['mu_n']['dim'])

        else:
            moms = self.Solid_1.getMoments(
                distrib=di_states['distrib']/self.scale)  # m**n

        di_states['mu_n'] = moms

        if name_unit == 'BatchCryst':
            rhos = rhos_susp
            h_in = None
            phis_in = None
        elif name_unit == 'SemibatchCryst' or name_unit == 'MSMPR':
            inlet_temp = u_input['Inlet']['temp']

            if self.Inlet.__module__ == 'PharmaPy.MixedPhases':
                rhos_in = self.Inlet.getDensity(temp=di_states['temp'])

                if 'distrib' in u_input['Inlet']:

                    inlet_distr = u_input['Inlet']['distrib']

                    mom_in = self.Inlet.Solid_1.getMoments(distrib=inlet_distr,
                                                            mom_num=3)
                elif 'mu_n' in u_input['Inlet']:

                    mom_in = np.array([u_input['Inlet']['mu_n'][3]])


                phi_in = 1 - self.Inlet.Solid_1.kv * mom_in
                phis_in = np.concatenate([phi_in, 1 - phi_in])

                h_in = self.Inlet.getEnthalpy(inlet_temp, phis_in, rhos_in)
            else:
                rho_liq_in = self.Inlet.getDensity(temp=inlet_temp)
                rho_sol_in = None

                rhos_in = np.array([rho_liq_in, rho_sol_in])
                h_in = self.Inlet.getEnthalpy(temp=inlet_temp)

                phis_in = [1, 0]

            rhos = [rhos_susp, rhos_in]

        # Balances
        material_bces, cryst_rate = self.material_balances(
            time, params, u_input, rhos, **di_states, phi_in=phis_in)

        if mat_bce:
            return material_bces
        elif enrgy_bce:
            energy_bce = self.energy_balances(
                time, params, cryst_rate, u_input, rhos, **di_states,
                h_in=h_in, heat_prof=True)

            return energy_bce

        else:

            if 'temp' in self.name_states:
                energy_bce = self.energy_balances(
                    time, params, cryst_rate, u_input, rhos, **di_states,
                    h_in=h_in)

                balances = np.append(material_bces, energy_bce)
            else:
                balances = material_bces

            self.derivatives = balances

            return balances

    def unit_jacobians(self, time, states, sens, params, fy, v_vector):
        if sens is not None:
            jac_states = self.jac_states_fun(time, states, params)
            jac_params = self.jac_params_fun(time, states, params)

            dsens_dt = np.dot(jac_states, sens) + jac_params

            if not isinstance(dsens_dt, np.ndarray):
                dsens_dt = dsens_dt._value

            return dsens_dt
        elif v_vector is not None:
            _, jac_v = self.jac_states_fun(time, states, params)(v_vector)

            return jac_v
        else:
            jac_states = self.jac_states_fun(time, states, params)

            if not isinstance(jac_states, np.ndarray):
                jac_states = jac_states._value

            return jac_states

    def jac_states_numerical(self, time, states, params, return_only=True):
        if return_only:
            return self.jac_states_vals
        else:
            def wrap_states(st): return self.unit_model(time, st, params)

            abstol = self.sundials_opt['atol']
            reltol = self.sundials_opt['rtol']
            jac_states = numerical_jac_central(wrap_states, states,
                                               dx=dx_jac_x,
                                               abs_tol=abstol, rel_tol=reltol)

            return jac_states

    def jac_params_numerical(self, time, states, params):
        def wrap_params(theta): return self.unit_model(time, states, theta)

        abstol = self.sundials_opt['atol']
        reltol = self.sundials_opt['rtol']
        p_bar = self.sundials_opt['pbar']

        dp = np.abs(p_bar) * np.sqrt(max(reltol, eps))

        jac_params = numerical_jac_central(wrap_params, params,
                                           dx=dp,
                                           abs_tol=abstol, rel_tol=reltol)

        return jac_params

    # def jac_states_ad(self, time, states, params):
    #     def wrap_states(st): return self.unit_model(time, st, params)
    #     jac_states = jacfwd(wrap_states)(states)

    #     return jac_states

    # def jac_params_ad(self, time, states, params):
    #     def wrap_params(theta): return self.unit_model(time, states, theta)
    #     jac_params = jacfwd(wrap_params)(params)

        return jac_params

    def rhs_sensitivity(self, time, states, sens, params):
        jac_params_vals = self.jac_params_fn(time, states, params)

        jac_states_vals = self.jac_states_fn(time, states, params,
                                             return_only=False)

        rhs_sens = np.dot(jac_states_vals, sens) + jac_params_vals

        self.jac_states_vals = jac_states_vals

        return rhs_sens

    def set_ode_problem(self, eval_sens, states_init, params_mergd,
                        jacv_prod):
        if eval_sens:
            problem = Explicit_Problem(self.unit_model, states_init,
                                       t0=self.elapsed_time,
                                       p0=params_mergd)

            if self.jac_type == 'finite_diff':
                self.jac_states_fn = self.jac_states_numerical
                self.jac_params_fn = self.jac_params_numerical

                problem.jac = self.jac_states_fn
                problem.rhs_sens = self.rhs_sensitivity

            elif self.jac_type == 'AD':
                self.jac_states_fn = self.jac_states_ad
                self.jac_params_fn = self.jac_params_ad

                problem.jac = self.jac_states_fn
                problem.rhs_sens = self.rhs_sensitivity

            elif self.jac_type == 'analytical':
                self.jac_states_fn = self.jac_states
                self.jac_params_fn = self.jac_params

                problem.jac = self.jac_states_fn
                problem.rhs_sens = self.rhs_sensitivity

            elif self.jac_type is None:
                pass
            else:
                raise NameError("Bad string value for the 'jac_type' argument")

        else:
            if self.state_event_list is None:
                def model(time, states, params=params_mergd):
                    return self.unit_model(time, states, params)

                problem = Explicit_Problem(model, states_init,
                                           t0=self.elapsed_time)
            else:
                sw0 = [True] * len(self.state_event_list)
                def model(time, states, sw=None):
                    return self.unit_model(time, states, params_mergd, sw)

                problem = Explicit_Problem(model, states_init,
                                           t0=self.elapsed_time, sw0=sw0)

            # ----- Jacobian callables
            if self.method == 'moments':
                # w.r.t. states
                # problem.jac = lambda time, states: \
                #     self.unit_jacobians(time, states, None, params_mergd,
                #                         None, None)

                pass

            elif self.method == 'fvm':
                # J*v product (AD, slower than the one used by SUNDIALS)
                if jacv_prod:
                    problem.jacv = lambda time, states, fy, v: \
                        self.unit_jacobians(time, states, None, params_mergd,
                                            fy, v)

        return problem

    def _eval_state_events(self, time, states, sw):
        events = eval_state_events(
            time, states, sw, self.len_states,
            self.states_uo, self.state_event_list, sdot=self.derivatives,
            discretized_model=False)

        return events

    def solve_unit(self, runtime=None, time_grid=None,
                   eval_sens=False,
                   jac_v_prod=False, verbose=True, test=False,
                   sundials_opts=None, any_event=True):
        """
        runtime : float (default = None)
            Value for the total unit runtime
        time_grid : list of float (optional, dafault = None)
            Optional list of time values for the integrator to use
            during simulation
        eval_sens : bool (optional, default = False)
            Boolean value indicating whether the parametric
            sensitivity system will be included during simulation.
            Must be True to access sensitivity information.
        jac_v_prod :
            TODO
        verbose : bool (optional, default = True)
            Boolean value indicating whether the simulator will
            output run statistics after simulation is complete.
            Use True if you want to see the number of function
            evaluations and wall-clock runtime for the unit.
        test :
            TODO
        sundials_opts :
            TODO
        any_event :
            TODO
        """

        if self.__class__.__name__ != 'BatchCryst':
            if self.method == 'moments':
                pass  # TODO: MSMPR MoM should be addressed?
            else:
                x_distr = getattr(self.Solid_1, 'x_distrib', [])
                self.states_in_dict['Inlet']['distrib'] = len(x_distr)

        self.Kinetics.target_idx = self.target_ind

        # ---------- Solid phase states
        if 'vol' in self.states_uo:
            if self.method == 'moments':
                init_solid = self.Solid_1.moments
                # exp = np.arange(0, self.Solid_1.num_mom) # TODO: problematic line for seeded crystallization.
                # init_solid = init_solid * (1e6)**exp

            elif self.method == '1D-FVM':
                x_grid = self.Solid_1.x_distrib
                init_solid = self.Solid_1.distrib * self.scale

        else:
            if self.method == 'moments':
                init_solid = self.Slurry.moments
                # exp = np.arange(0, self.Solid_1.num_mom) # TODO
                # init_solid = init_solid * (1e6)**exp

            elif self.method == '1D-FVM':
                x_grid = self.Slurry.x_distrib
                init_solid = self.Slurry.distrib * self.scale

        self.dx = self.Slurry.dx
        self.x_grid = self.Slurry.x_distrib

        # ---------- Liquid phase states
        init_liquid = self.Liquid_1.mass_conc.copy()

        self.num_species = len(init_liquid)

        self.len_states = [self.num_distr, self.num_species]  # TODO: not neces

        if 'vol' in self.states_uo:  # Batch or semibatch
            vol_init = self.Slurry.getTotalVol()
            init_susp = np.append(init_liquid, vol_init)

            self.len_states.append(1)
        else:
            init_susp = init_liquid

        if self.reset_states:
            self.reset()

        # ---------- Read time
        if runtime is not None:
            final_time = runtime + self.elapsed_time

        if time_grid is not None:
            final_time = time_grid[-1]

        if self.scale_flag:
            self.scale_flag = False

        states_init = np.append(init_solid, init_susp)

        if self.vol_tank is None:
            if isinstance(self, SemibatchCryst):
                time_vec = np.linspace(self.elapsed_time, final_time)
                vol_flow = self.get_inputs(time_vec)['Inlet']['vol_flow']

                self.vol_tank = trapezoidal_rule(time_vec, vol_flow)

            else:
                self.vol_tank = self.Slurry.vol

        self.diam_tank = (4/np.pi * self.vol_tank)**(1/3)
        self.area_base = np.pi/4 * self.diam_tank**2
        self.vol_tank *= 1 / self.vol_offset

        if 'temp_ht' in self.states_uo:

            if len(self.profiles_runs) == 0:
                temp_ht = self.Liquid_1.temp
            else:
                temp_ht = self.profiles_runs[-1]['temp_ht'][-1]

            states_init = np.concatenate(
                (states_init, [self.Liquid_1.temp, temp_ht]))

            self.len_states += [1, 1]
        elif 'temp' in self.states_uo:
            states_init = np.append(states_init, self.Liquid_1.temp)
            self.len_states += [1]

        merged_params = self.Kinetics.concat_params()[self.mask_params]

        # ---------- Create problem
        problem = self.set_ode_problem(eval_sens, states_init,
                                       merged_params, jac_v_prod)

        self.derivatives = problem.rhs(self.elapsed_time, states_init,
                                       merged_params)

        if len(self.state_event_list) > 0:
            def new_handle(solver, info):
                return handle_events(solver, info, self.state_event_list,
                                     any_event=any_event)

            problem.state_events = self._eval_state_events
            problem.handle_event = new_handle

        # ---------- Set solver
        # General
        solver = CVode(problem)
        solver.iter = 'Newton'
        solver.discr = 'BDF'

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        self.sundials_opt = solver.get_options()

        if eval_sens:
            solver.sensmethod = 'SIMULTANEOUS'
            solver.suppress_sens = False
            solver.report_continuously = True

        if self.method == '1D-FVM':
            solver.linear_solver = 'SPGMR'  # large, sparse systems

        if not verbose:
            solver.verbosity = 50

        # ---------- Solve model
        time, states = solver.simulate(final_time, ncp_list=time_grid)

        self.retrieve_results(time, states)

        # ---------- Organize sensitivity
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

    def paramest_wrapper(self, params, t_vals,
                         modify_phase=None, modify_controls=None,
                         scale_factor=1e-3, run_args={}, reord_sens=True):
        self.reset()
        self.params_iter = params

        self.Kinetics.set_params(params)

        self.elapsed_time = 0

        if isinstance(modify_phase, dict) and len(modify_phase) > 0:

            if 'Liquid' not in modify_phase and 'Solid' not in modify_phase:
                raise ValueError(
                    "Phase modifier must specify the targeted phase, i.e. "
                    "must have an additional layer with keys 'Liquid_1' "
                    "and/or 'Solid_1'")

            liquid_mod = modify_phase.get('Liquid', {})
            solid_mod = modify_phase.get('Solid', {})

            self.Liquid_1.updatePhase(**liquid_mod)
            self.Solid_1.updatePhase(**solid_mod)

        if isinstance(modify_controls, dict):
            for key, val in modify_controls.items():
                self.controls[key].update(val)

        if self.param_wrapper is None:
            if self.method == 'moments':
                t_prof, states, sens = self.solve_unit(time_grid=t_vals,
                                                       eval_sens=True,
                                                       verbose=False,
                                                       **run_args)

                if reord_sens:
                    sens = reorder_sens(sens, separate_sens=False)
                else:
                    sens = np.stack(sens)

                result = (states, sens)
            else:
                t_prof, states_out = self.solve_unit(time_grid=t_vals,
                                                     eval_sens=False,
                                                     verbose=False,
                                                     **run_args)

                result = states_out

        elif callable(self.param_wrapper):
            if self.method == 'moments':
                t_prof, states, sens = self.solve_unit(time_grid=t_vals,
                                                       eval_sens=True,
                                                       verbose=False,
                                                       **run_args)

                # dy/dt for each state separately
                sens_sep = reorder_sens(sens, separate_sens=True)

                # TODO: is this the better way of naming the states?
                di_keys = ['mu_%s' % ind for ind in range(self.num_distr)]
                di_keys += ['w_%s' % name for name in self.name_species]
                di_keys.append('vol')

                sens_sep = dict(zip(di_keys, sens_sep))

                result = self.param_wrapper(self.result, sens_sep,
                                            reord_sens=reord_sens)

        return result

    def flatten_states(self):
        out = flatten_states(self.profiles_runs)

        return out

    def plot_profiles(self, **fig_kwargs):
        """

        Parameters
        ----------
        fig_kwargs : keyword arguments
            keyword arguments to be passed to the plot.subplots() method

        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.

        """

        def get_mu_labels(mu_idx, msmpr=False):
            out = []
            for idx in mu_idx:
                name = '$\mu_{%i}$' % idx

                if idx == 0:
                    unit = '#'
                elif idx == 1:
                    unit = 'm'
                else:
                    unit = '$\mathrm{m^{%i}}$' % idx

                if msmpr:
                    unit += ' $\mathrm{m^{-3}}$'

                unit = r' (%s)' % unit

                out.append(name + unit)

            return out

        states = [('mu_n', (0, )), 'temp', ('mass_conc', (self.target_ind,)),
                  'supersat']

        figmap = [0, 4, 5, 5]
        ylabels = ['mu_0', 'T', 'C_j', 'sigma']

        if hasattr(self.result, 'temp_ht'):
            states.append('temp_ht')
            figmap.append(4)
            ylabels.append('T_{ht}')

        fig, ax = plot_function(self, states, fig_map=figmap,
                                nrows=3, ncols=2, ylabels=ylabels,
                                **fig_kwargs)

        ax[0, 0].legend().remove()

        time = self.result.time
        moms = self.result.mu_n

        is_msmpr = self.__class__.__name__ == 'MSMPR'
        labels_moms = get_mu_labels(range(moms.shape[1]), msmpr=is_msmpr)

        for ind, row in enumerate(moms[:, 1:].T):
            ax.flatten()[ind + 1].plot(time, row)

        for ind, lab in enumerate(labels_moms):
            ax.flatten()[ind].set_ylabel(lab)

        # Solubility
        ax[2, 1].plot(time, self.result.solubility)
        ax[2, 1].lines[1].set_color('k')
        ax[2, 1].lines[1].set_alpha(0.4)

        ax[2, 1].legend([self.target_comp[0], 'solubility'])

        fig.tight_layout()
        return fig, ax

    def plot_csd(self, times=(0,), logy=False, vol_based=False, **fig_kw):

        if vol_based:
            state_plot = ['vol_distrib']
            y_lab = ('f_v', )
        else:
            state_plot = ['distrib']
            y_lab = ('f', )

        fig, axis = plot_distrib(self, state_plot, times=times,
                                 x_name='x_cryst', ylabels=y_lab, legend=False,
                                 **fig_kw)

        # axis.set_xlabel('$x$ ($\mathregular{\mu m}$)')
        axis.set_xscale('log')

        fig.texts[0].remove()
        axis.set_xlabel('$x$ ($\mathregular{\mu m}$)')

        return fig, axis

    def plot_csd_heatmap(self, vol_based=False, **fig_kw):
        self.flatten_states()

        if self.method != '1D-FVM':
            raise RuntimeError('No 3D data to show. Run crystallizer with the '
                               'FVM method')

        res = self.result
        x_mesh, t_mesh = np.meshgrid(res.x_cryst, res.time)

        fig, ax = plt.subplots(**fig_kw)

        if vol_based:
            distrib = res.vol_distrib.T
        else:
            distrib = res.distrib.T

        cf = ax.contourf(x_mesh.T, t_mesh.T, distrib, cmap=cm.Blues,
                         levels=150)

        cbar = fig.colorbar(cf)

        if self.scale == 1:
            cbar.ax.set_ylabel(r'$f$ $\left( \frac{\#}{m^3 \mu m} \right)$')
        else:
            exp = int(np.log10(self.scale))
            cbar.ax.set_ylabel(
                r'$f$ $\times 10^{%i}$ $\left( \frac{\#}{m^3 \mu m} \right)$ ' % exp)

        # Edit
        ax.set_xlabel(r'size ($\mu m$)')
        ax.set_ylabel('time (s)')
        ax.invert_yaxis()

        ax.set_xscale('log')

        return fig, ax

    def plot_sens(self, mode='per_parameter'):
        if type(self.timeProf) is list:
            self.flatten_states()

        if self.sensit is None:
            raise AttributeError("No sensitivities detected. Run the unit "
                                 " with 'eval_sens'=True")

        if mode == 'per_parameter':
            sens_data = self.sensit
        elif mode == 'per_state':
            sens_data = reorder_sens(self.sensit, separate_sens=True)

        # Name states
        name_mom = ['\mu_%i' % i for i in range(self.num_distr)]
        name_conc = ["C_{" + self.name_species[ind] + "}"
                     for ind in range(len(self.Liquid_1.name_species))]

        name_others = []
        if 'vol' in self.states_uo:
            name_others.append('vol')

        if 'temp' in self.states_uo:
            name_others.append('temp')

        name_states = name_mom + name_conc + name_others
        name_params = [name for ind, name in
                       enumerate(self.Kinetics.name_params)
                       if self.mask_params[ind]]

        fig, axis = plot_sens(self.result.time, sens_data,
                              name_states=name_states,
                              name_params=name_params,
                              mode=mode)

        return fig, axis

    def animate_cryst(self, filename=None, fps=5, step_data=1):
        from matplotlib.animation import FuncAnimation
        from matplotlib.animation import FFMpegWriter

        if type(self.timeProf) is list:
            self.flatten_states()

        if filename is None:
            filename = 'anim'

        fig_anim, ax_anim = plt.subplots(figsize=(5, 3.125))

        # ax_anim.set_xlim(0, self.x_grid.max())
        ax_anim.set_xlabel(r'crystal size ($\mu m$)')

        ax_anim.set_ylabel('counts')

        def func_data(ind):
            dist = self.distribProf[ind]
            return dist

        line, = ax_anim.plot(self.x_grid, func_data(0), '-o', mfc='None',
                             ms='2')
        time_tag = ax_anim.text(
            1, 1.04, '$time = {:.1f}$ s'.format(self.timeProf[0]),
            horizontalalignment='right',
            transform=ax_anim.transAxes)

        def func_anim(ind):
            f_vals = func_data(ind)
            line.set_ydata(f_vals)
            plt.gca().set_xscale("log")

            if f_vals.max() > f_vals.min():
                ax_anim.set_ylim(f_vals.min()*1.15, f_vals.max()*1.15)

            time_tag.set_text('$time = {:.1f}$ s'.format(self.timeProf[ind]))

            fig_anim.tight_layout()

        frames = np.arange(0, len(self.timeProf), step_data)
        animation = FuncAnimation(fig_anim, func_anim, frames=frames,
                                  repeat=True)

        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'),
                              bitrate=1800)

        animation.save(filename + '.mp4', writer=writer)

        return animation, fig_anim, ax_anim


class BatchCryst(_BaseCryst):
    """Construct a Batch Crystallizer object
    
    Parameters
    ----------
    target_comp : str, list of strings
        Name of the crystallizing compound(s) from .json file.
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computation
    method : str
        Choice of the numerical method. Options are: 'moments', '1D-FVM'
    scale : float
        Scaling factor by which crystal size distribution will be
        multiplied.
    controls : dict of dicts (funcs) (optional, default = None)
        Dictionary with keys representing the state (e.g.'Temp')
        which is controlled and the value indicating the function
        to use while computing the varible. Functions are of the form
        f(time) = state_value
    cfun_solub: callable
        User defined function for the solubility function :
        func(conc)
    adiabatic : bool (optional, default =True)
        Boolean value indicating whether the heat transfer of
        the crystallization is considered.
    rad_zero : float (optional)
        size of the first bin of the CSD discretization [m]
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be
        reset before simulation
    basis : str (optional, default = 'mass_conc')
        Options : 'massfrac', 'massconc'
    state_events : lsit of dict(s)
        list of dictionaries, each one containing the specification of a
        state event
    """

    def __init__(self, target_comp, mask_params=None,
                 method='1D-FVM', scale=1, vol_tank=None,
                 controls=None, adiabatic=False,
                 rad_zero=0, reset_states=False,
                 h_conv=1000, vol_ht=None, basis='mass_conc',
                 jac_type=None, state_events=None, param_wrapper=None):


        super().__init__(mask_params, method, target_comp, scale, vol_tank,
                         controls, adiabatic,
                         rad_zero, reset_states, h_conv, vol_ht,
                         basis, jac_type, state_events, param_wrapper)

        self.is_continuous = False
        self.oper_mode = 'Batch'

        self.vol_offset = 0.75

    def jac_states(self, time, states, params, return_only=True):

        if return_only:
            return self.jac_states_vals
        else:
            # Name states
            vol_liq = states[-1]

            num_material = self.num_distr + self.num_species
            w_conc = states[self.num_distr:num_material]

            control = self.controls['temp']
            temp = control['fun'](time, *control['args'], **control['kwargs'])

            num_states = len(states)
            conc_tg = w_conc[self.target_ind]
            c_sat = self.Kinetics.get_solubility(temp, w_conc)

            moms = states[:self.num_distr]
            idx_moms = np.arange(1, len(moms))

            rho_l = self.Liquid_1.getDensity(temp=temp)
            rho_c = self.Solid_1.getDensity(temp=temp)
            kv = self.Solid_1.kv

            # Kinetics
            b_pr = self.Kinetics.prim_nucl
            b_sec = self.Kinetics.sec_nucl

            nucl = b_pr + b_sec
            gr = self.Kinetics.growth

            g_exp = self.Kinetics.params['growth'][-1]
            bp_exp = self.Kinetics.params['nucl_prim'][-1]
            k_s, _, bs_exp, bs2_exp = self.Kinetics.params['nucl_sec']
            # bs2_exp = self.Kinetics.params['nucl_sec'][-1]

            jacobian = np.zeros((num_states, num_states))

            # ----- Moments columns
            wrt_mu = idx_moms * gr

            rng = np.arange(len(wrt_mu))
            jacobian[rng + 1, rng] = wrt_mu

            # dfu0_dmu3
            jacobian[0, self.num_distr - 1] = vol_liq * bs2_exp * b_sec / \
                (moms[3] + eps)

            # ssat = (conc_tg - c_sat) / c_sat

            # # bsec_other = k_s * ssat**bs_exp * (kv * moms[3])**bs2_exp
            # dfu0_dmu3 = k_s * ssat**bs_exp * bs2_exp * kv * \
            #     (kv * moms[3])**(bs2_exp - 1)

            # jacobian[0, self.num_distr - 1] = dfu0_dmu3

            # Second moment column (concentration eqns)
            dtr_mu2 = 3 * kv * gr * rho_c * \
                (1e-6)**3  # factor from material bce

            dfconc_dmu2 = -1/vol_liq * dtr_mu2 * (self.kron_jtg - w_conc/rho_l)
            jacobian[self.num_distr:self.num_distr + dfconc_dmu2.shape[0],
                     2] = dfconc_dmu2

            # Volume eqn
            jacobian[-1, 2] = -dtr_mu2 / rho_l  # dfvol/dmu2

            # ----- Concentration columns
            # Moment eqns
            conc_diff = conc_tg - c_sat

            dfmu0_dconc = (bp_exp * b_pr + bs_exp * b_sec) * \
                vol_liq / conc_diff
            dfmun_dconc = idx_moms * moms[:-1] * g_exp/conc_diff * gr

            jacobian[0, self.num_distr +
                     self.target_ind] = dfmu0_dconc

            jacobian[
                1:1 + dfmun_dconc.shape[0],
                self.num_distr + self.target_ind] = dfmun_dconc

            # Concentration eqns
            tr = 3 * kv * gr * moms[2] * rho_c * (1e-6)**3
            dtr_dconc_tg = g_exp * tr / conc_diff

            first_conc = np.outer(self.kron_jtg - w_conc/rho_l, self.kron_jtg)
            second_conc = tr/rho_l * np.eye(len(w_conc))

            dfconc_dconc = -1/vol_liq * \
                (dtr_dconc_tg * first_conc + second_conc)

            jacobian[self.num_distr:-1, self.num_distr:-1] = dfconc_dconc

            # Volume eqn
            jacobian[-1, self.num_distr + self.target_ind] = - \
                dtr_dconc_tg / rho_l

            # ----- Volume column
            # mu_zero eqn
            jacobian[0, -1] = nucl  # dfmu_0/dvol

            # Concentration eqn
            dfconc_dvol = 1/vol_liq**2 * (self.kron_jtg*tr - w_conc/rho_l * tr)
            jacobian[self.num_distr:self.num_distr + dfconc_dvol.shape[0],
                     -1] = dfconc_dvol

            return jacobian

    def jac_params(self, time, states, params):

        state_di = unpack_states(states, self.dim_states, self.name_states)

        control = self.controls['temp']
        temp = control['fun'](time, *control['args'], **control['kwargs'])

        num_states = len(states)

        vol_liq = states[-1]
        moms = states[:self.num_distr]
        num_material = self.num_distr + self.num_species
        w_conc = states[self.num_distr:num_material]
        conc_tg = w_conc[self.target_ind]

        kv = self.Solid_1.kv
        rho_c = self.Solid_1.getDensity(temp=temp)
        rho_l = self.Liquid_1.getDensity(temp=temp)

        b_sec = self.Kinetics.sec_nucl

        dbp, dbs, dg, _, _ = self.Kinetics.deriv_cryst(conc_tg, w_conc, temp)
        dbs_ds2 = b_sec * np.log(max(eps, kv * moms[3]*1e-18))
        dbs = np.append(dbs, dbs_ds2)

        # dg *= 1e-6  # to m/s

        num_bp = len(dbp)
        num_bs = len(dbs)
        num_nucl = len(dbp) + len(dbs)
        num_gr = len(dg)

        # TODO: the 3 is only to account for dissolution
        num_params = num_nucl + num_gr + 3

        idx_moms = np.arange(1, self.num_distr)
        g_section = np.outer(idx_moms * moms[:-1], dg)

        # ----- Moment equations
        jacobian = np.zeros((num_states, num_params))

        # Zeroth moment eqn
        jacobian[0, :num_bp] = vol_liq * dbp
        jacobian[0, num_bp:num_bp + num_bs] = vol_liq * dbs

        # jacobian[0] *= vol_liq

        # 1 and higher order moments eqns
        jacobian[1:1 + g_section.shape[0],
                 num_nucl:num_nucl + g_section.shape[1]] = g_section

        # ----- Concentration eqns
        dtr_g = 3 * kv * rho_c * moms[2] * dg * \
            (1e-6)**3  # factor from material bce
        dconc_dg = -1/vol_liq * np.outer(self.kron_jtg - w_conc/rho_l, dtr_g)

        jacobian[self.num_distr:self.num_distr + dconc_dg.shape[0],
                 num_nucl:num_nucl + dconc_dg.shape[1]] = dconc_dg

        # ----- Volume eqn
        jacobian[-1, num_nucl:num_nucl + dtr_g.shape[0]] = -dtr_g / rho_l

        return jacobian[:, self.mask_params]

    def material_balances(self, time, params, u_inputs, rhos, mu_n,
                          distrib, mass_conc, temp, temp_ht, vol, phi_in=None):

        # 'vol' represents liquid volume

        rho_liq, rho_s = rhos

        vol_solid = mu_n[3] * self.Solid_1.kv  # mu_3 is total, not by volume
        vol_slurry = vol + vol_solid

        if self.method == 'moments':
            ddistr_dt, transf = self.method_of_moments(distrib, mass_conc, temp,
                                                       params, rho_s,
                                                       vol=vol_slurry)
        elif self.method == '1D-FVM':
            ddistr_dt, transf = self.fvm_method(distrib, mu_n, mass_conc, temp,
                                                params, rho_s, vol=vol_slurry)

        # Balance for target
        self.Liquid_1.updatePhase(mass_conc=mass_conc, vol=vol)

        dvol_liq = -transf/rho_liq  # TODO: results not consistent with mu_3
        dcomp_dt = -transf/vol * (self.kron_jtg - mass_conc/rho_liq)

        dliq_dt = np.append(dcomp_dt, dvol_liq)

        if self.basis == 'mass_frac':
            dcomp_dt *= 1 / rho_liq

        dmaterial_dt = np.concatenate((ddistr_dt, dliq_dt))

        return dmaterial_dt, transf

    def energy_balances(self, time, params, cryst_rate, u_inputs, rhos,
                        mu_n, distrib, mass_conc, temp, temp_ht, vol,
                        h_in=None, heat_prof=False):

        vol_solid = mu_n[3] * self.Solid_1.kv  # mu_3 is total, not by volume
        vol_total = vol + vol_solid

        phi = vol / vol_total
        phis = [phi, 1 - phi]

        # Suspension properties  TODO: slurry should be updated here
        capacitance = self.Slurry.getCp(temp, phis, rhos,
                                        times_vliq=True)

        # Renaming
        dh_cryst = -1.46e4  # J/kg
        # dh_cryst = -self.Liquid_1.delta_fus[self.target_ind] / \
        #     self.Liquid_1.mw[self.target_ind] * 1000  # J/kg

        vol = vol / phi

        height_liq = vol / (np.pi/4 * self.diam_tank**2)
        area_ht = np.pi * self.diam_tank * height_liq + self.area_base  # m**2

        source_term = dh_cryst*cryst_rate

        if self.adiabatic:
            ht_term = 0
        elif 'temp' in self.controls.keys():
            ht_term = capacitance * vol  # return capacitance
        elif 'temp' in self.states_uo:
            ht_term = self.u_ht*area_ht*(temp - temp_ht)

        if heat_prof:
            heat_components = np.hstack([source_term, ht_term])
            return heat_components
        else:
            # Balance inside the tank
            dtemp_dt = (-source_term - ht_term) / capacitance / vol

            if temp_ht is not None:
                ht_dict = self.Utility.get_inputs(time)
                tht_in = ht_dict['temp_in']
                flow_ht = ht_dict['vol_flow']

                cp_ht = 4180  # J/kg/K
                rho_ht = 1000
                vol_ht = vol*0.14  # m**3

                dtht_dt = flow_ht / vol_ht * (tht_in - temp_ht) - \
                    self.u_ht*area_ht*(temp_ht - temp) / rho_ht/vol_ht/cp_ht

                return dtemp_dt, dtht_dt

            else:
                return dtemp_dt

    def retrieve_results(self, time, states):
        time = np.array(time)
        self.elapsed_time = time[-1]

        # ---------- Create result object
        dp = unpack_states(states, self.dim_states, self.name_states)
        dp['time'] = time

        if self.method == '1D-FVM':
            dp['distrib'] *= 1 / self.scale
            dp['x_cryst'] = self.x_grid

            moms = self.Solid_1.getMoments(distrib=dp['distrib'])
            dp['mu_n'] = moms

            dp['vol_distrib'] = self.Solid_1.convert_distribution(
                num_distr=dp['distrib'])

        if 'temp' in self.controls:
            control = self.controls['temp']
            dp['temp'] = control['fun'](time, *control['args'], **control['kwargs'])

        sat_conc = self.Kinetics.get_solubility(dp['temp'], dp['mass_conc'])

        supersat = dp['mass_conc'][:, self.target_ind] - sat_conc

        dp['solubility'] = sat_conc
        dp['supersat'] = supersat

        self.profiles_runs.append(dp)
        dp = self.flatten_states()

        if self.method == 'moments':
            dp['mu_n'] = dp['mu_n'] * (1e-6)**np.arange(self.num_distr)

        self.result = DynamicResult(self.states_di, self.fstates_di,
                                            **dp)
        # ---------- Update phases
        vol_sol = dp['mu_n'][-1, 3] * self.Solid_1.kv

        rho_solid = self.Solid_1.getDensity()
        mass_sol = rho_solid * vol_sol

        vol_slurry = dp['vol'][-1] + vol_sol

        self.Liquid_1.updatePhase(mass_conc=dp['mass_conc'][-1],
                                  vol=dp['vol'][-1])

        self.Liquid_1.temp = dp['temp'][-1]
        self.Solid_1.temp = dp['temp'][-1]
        slurry = Slurry(vol=vol_slurry)
        if self.method == '1D-FVM':
            self.Solid_1.updatePhase(distrib=dp['distrib'][-1],
                                     mass=mass_sol)

        elif self.method == 'moments':
            self.Solid_1.updatePhase(moments=dp['mu_n'][-1])

        # Create outlets
        liquid_out = copy.deepcopy(self.Liquid_1)
        solid_out = copy.deepcopy(self.Solid_1)

        self.Outlet = slurry
        self.Outlet.Phases = (liquid_out, solid_out)

        self.outputs = dp

        # ---------- Calculate heat duty
        self.get_heat_duty(time, states)

    def get_heat_duty(self, time, states):
        q_heat = np.zeros((len(time), 2))

        if self.params_iter is None:
            merged_params = self.Kinetics.concat_params()[self.mask_params]
        else:
            merged_params = self.params_iter

        for ind, row in enumerate(states):
            row = row.copy()
            row[:self.num_distr] *= self.scale  # scale distribution
            q_heat[ind] = self.unit_model(time[ind], row, merged_params,
                                          enrgy_bce=True)

        if 'temp' in self.controls.keys():
            q_gen, capacitance = q_heat.T

            dT_dt = np.diff(self.result.temp) / \
                np.diff(self.result.time)

            q_instant = dT_dt * capacitance[1:] + q_gen[1:]
            time = time[1:]

        else:
            q_instant = q_heat  # TODO: write for other scenarios

        self.heat_duty = np.array([0, trapezoidal_rule(time, q_instant)])
        self.duty_type = [0, -2]


class MSMPR(_BaseCryst):
    """ Construct a MSMPR object.
    
    Parameters
    ----------
    target_comp : str, list of strings
        Name of the crystallizing compound(s) from .json file.
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computation
    method : str
        Choice of the numerical method. Options are: 'moments', '1D-FVM'
    scale : float
        Scaling factor by which crystal size distribution will be
        multiplied.
    controls : dict of dicts (funcs) (optional, default = None)
        Dictionary with keys representing the state (e.g.'Temp')
        which is controlled and the value indicating the function
        to use while computing the varible. Functions are of the form
        f(time) = state_value
    cfun_solub: callable
        User defined function for the solubility function :
        func(conc)
    adiabatic : bool (optional, default =True)
        Boolean value indicating whether the heat transfer of
        the crystallization is considered.
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be
        reset before simulation
    basis : str (optional, default = 'mass_conc')
        Options : 'massfrac', 'massconc'
    state_events : lsit of dict(s)
        list of dictionaries, each one containing the specification of a
        state event
    """
    def __init__(self, target_comp,
                 mask_params=None,
                 method='1D-FVM', scale=1, vol_tank=None,
                 controls=None, adiabatic=False, rad_zero=0,
                 reset_states=False,
                 h_conv=1000, vol_ht=None, basis='mass_conc',
                 jac_type=None, num_interp_points=3, state_events=None,
                 param_wrapper=None):

        super().__init__(mask_params, method, target_comp, scale, vol_tank,
                         controls, adiabatic, rad_zero,
                         reset_states, h_conv, vol_ht,
                         basis, jac_type, state_events, param_wrapper)

        # self.states_uo.append('conc_j')
        self.is_continuous = True
        self.oper_mode = 'Continuous'
        self._Inlet = None

        # self.nomenclature()

        self.vol_offset = 0.75
        self.num_interp_points = num_interp_points

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet_object):
        self._Inlet = inlet_object
        self._Inlet.num_interpolation_points = self.num_interp_points

    def _get_tau(self):
        time_upstream = getattr(self.Inlet, 'time_upstream', None)
        if time_upstream is None:
            time_upstream = [0]

        inputs = self.get_inputs(time_upstream[-1])

        volflow_in = inputs['Inlet']['vol_flow']
        tau = self.Liquid_1.vol / volflow_in

        self.tau = tau
        return tau

    def solve_steady_state(self, frac_seed, temp):

        vol = self.vol_slurry
        flow_v = self.Inlet.vol_flow / vol  # 1/s

        x_vec = self.Solid_1.x_distrib  # um

        kv = self.Solid_1.kv
        rho = self.Solid_1.getDensity(temp=temp)

        w_in = self.Inlet.Liquid_1.mass_frac[self.target_ind]

        def fun_of_frac(w_tank, full_output=False):

            nucl, growth, _ = self.Kinetics.get_kinetics(w_tank, temp, kv)

            # growth *= 1e-6

            # Analytical solution to f(x)
            f_zero = nucl / growth  # num / s / m**3 / um
            f_x = np.exp(-flow_v / growth * x_vec) * f_zero

            # vfrac_ph = self.Slurry.getFractions(f_x)
            # rho_liq = self.Liquid_1.getDensity(temp=temp)

            mu_2 = trapezoidal_rule(x_vec, x_vec**2 * f_x)

            kinetic_term = -3 * kv * rho * growth * mu_2
            flow_term = flow_v * (w_in - w_tank)

            conc_eqn = kinetic_term + flow_term

            if full_output:
                return f_x, conc_eqn
            else:
                return conc_eqn

        # Solve eqn
        # frac_seed = self.Liquid_1.mass_frac[self.target_ind]
        w_convg, info = newton(fun_of_frac, frac_seed, full_output=True)
        f_convg, final_fn = fun_of_frac(w_convg, full_output=True)

        return x_vec, f_convg, w_convg, info, final_fn

    def material_balances(self, time, params, u_inputs, rhos, mu_n,
                          distrib, mass_conc, temp, temp_ht, vol, phi_in):

        rho_sol = rhos[0][1]

        input_flow = u_inputs['Inlet']['vol_flow']

        input_conc = u_inputs['Liquid_1']['mass_conc']

        if self.method == 'moments':
            input_distrib = u_inputs['Inlet']['mu_n'] * (1e6)**np.arange(self.num_distr)#* self.scale
            ddistr_dt, transf = self.method_of_moments(distrib, mass_conc, temp,
                                                       params, rho_sol)
        elif self.method == '1D-FVM':
            input_distrib = u_inputs['Inlet']['distrib'] * self.scale
            ddistr_dt, transf = self.fvm_method(distrib, mu_n, mass_conc, temp,
                                                params, rho_sol)

            self.Solid_1.moments[[2, 3]] = mu_n[[2, 3]]

        # ---------- Add flow terms
        # Distribution
        tau_inv = input_flow / vol
        flow_distrib = tau_inv * (input_distrib - distrib)

        ddistr_dt = ddistr_dt + flow_distrib
        # Liquid phase
        phi = 1 - self.Solid_1.kv * mu_n[3]

        c_tank = mass_conc

        flow_term = tau_inv * (input_conc*phi_in[0] - c_tank*phi)
        transf_term = transf * (self.kron_jtg - c_tank / rho_sol)
        dcomp_dt = 1 / phi * (flow_term - transf_term)

        if self.basis == 'mass_frac':
            rho_liq = self.Liquid_1.getDensity()
            dcomp_dt *= 1 / rho_liq

        dmaterial_dt = np.concatenate((ddistr_dt, dcomp_dt))

        return dmaterial_dt, transf

    def energy_balances(self, time, params, cryst_rate, u_inputs, rhos, mu_n,
                        distrib, mass_conc, temp, temp_ht, vol,
                        h_in, heat_prof=False):

        rho_susp, rho_in = rhos

        input_flow = u_inputs['Inlet']['vol_flow']

        # Thermodynamic properties (basis: slurry volume)
        phi_liq = 1 - self.Solid_1.kv * mu_n[3]

        phis = [phi_liq, 1 - phi_liq]
        h_sp = self.Slurry.getEnthalpy(temp, phis, rho_susp)
        capacitance = self.Slurry.getCp(temp, phis, rho_susp)  # J/m**3/K

        # Renaming
        dh_cryst = -1.46e4  # J/kg  # TODO: read this from json file
        # dh_cryst = -self.Liquid_1.delta_fus[self.target_ind] / \
        #     self.Liquid_1.mw[self.target_ind] * 1000  # J/kg

        height_liq = vol / (np.pi/4 * self.diam_tank**2)
        area_ht = np.pi * self.diam_tank * height_liq + self.area_base  # m**2

        # Energy terms (W)
        flow_term = input_flow * (h_in - h_sp)
        source_term = dh_cryst*cryst_rate * vol

        if 'temp' in self.controls.keys():
            ht_term = capacitance * vol  # return capacitance
        elif 'temp' in self.states_uo:
            ht_term = self.u_ht*area_ht*(temp - temp_ht)
        if heat_prof:
            heat_components = np.hstack([source_term, ht_term, flow_term])
            return heat_components
        else:
            # Balance inside the tank
            dtemp_dt = (flow_term - source_term - ht_term) / vol / capacitance

            # Balance in the jacket
            ht_media = self.Utility.get_inputs(time)
            flow_ht = ht_media['vol_flow']
            tht_in = ht_media['temp_in']

            cp_ht = self.Utility.cp
            rho_ht = self.Utility.rho

            vol_ht = self.vol_tank*0.14  # m**3

            dtht_dt = flow_ht / vol_ht * (tht_in - temp_ht) - \
                self.u_ht*area_ht*(temp_ht - temp) / rho_ht/vol_ht/cp_ht

            return dtemp_dt, dtht_dt

    def retrieve_results(self, time, states):
        time = np.array(time)

        # ---------- Create result object
        inputs = self.get_inputs(time)
        volflow = inputs['Inlet']['vol_flow']

        dp = unpack_states(states, self.dim_states, self.name_states)

        dp['time'] = time
        dp['vol_flow'] = volflow
        dp['x_cryst'] = self.x_grid

        if 'temp' in self.controls:
            control = self.controls['temp']
            dp['temp'] = control['fun'](time, *control['args'], **control['kwargs'])

        sat_conc = self.Kinetics.get_solubility(dp['temp'], dp['mass_conc'])

        supersat = dp['mass_conc'][:, self.target_ind] - sat_conc

        dp['solubility'] = sat_conc
        dp['supersat'] = supersat

        vol_slurry = self.Slurry.vol

        if self.method == '1D-FVM':
            dp['distrib'] *= 1 / self.scale
            moms = self.Solid_1.getMoments(distrib=dp['distrib'])
            dp['mu_n'] = moms

            dp['vol_distrib'] = self.Solid_1.convert_distribution(
                num_distr=dp['distrib'])

            self.Solid_1.updatePhase(distrib=dp['distrib'][-1] * vol_slurry)

        if self.method == 'moments':
            dp['mu_n'] = dp['mu_n'] * (1e-6)**np.arange(self.num_distr)

        if self.__class__.__name__ == 'SemibatchCryst':
            dp['total_distrib'] = dp['distrib']

        self.profiles_runs.append(dp)
        dp = self.flatten_states()

        self.outputs = dp

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        # ---------- Update phases

        self.Solid_1.temp = dp['temp'][-1]
        self.Liquid_1.temp = dp['temp'][-1]

        if type(self) == MSMPR:
            vol_slurry = self.Slurry.vol
            vol_liq = (1 - self.Solid_1.kv * dp['mu_n'][-1, 3]) * vol_slurry

            self.Liquid_1.updatePhase(vol=vol_liq,
                                      mass_conc=dp['mass_conc'][-1])
            if self.method == '1D-FVM':
                distrib_tilde = dp['distrib'][-1] * vol_slurry
                self.Solid_1.updatePhase(distrib=distrib_tilde)

                self.Slurry = Slurry()

            elif self.method == 'moments':
                self.Slurry = Slurry(moments=dp['mu_n'][-1], vol=vol_slurry)

        else:
            vol_liq = dp['vol'][-1]

            rho_solid = self.Solid_1.getDensity()
            vol_solid = dp['mu_n'][-1, 3] * self.Solid_1.kv * rho_solid

            vol_slurry = vol_solid + vol_liq

            if self.method == '1D-FVM':
                distrib_tilde = dp['total_distrib'][-1]
                self.Solid_1.updatePhase(distrib=distrib_tilde)

                self.Slurry = Slurry()

            elif self.method == 'moments':
                pass  # TODO

        self.Slurry.Phases = (self.Solid_1, self.Liquid_1)
        self.elapsed_time = time[-1]

        # ---------- Create output stream
        path = self.Liquid_1.path_data

        solid_comp = np.zeros(self.num_species)
        solid_comp[self.target_ind] = 1

        if type(self) == MSMPR:
            liquid_out = LiquidStream(path,
                                      mass_conc=dp['mass_conc'][-1],
                                      temp=dp['temp'][-1], check_input=False)

            solid_out = SolidStream(path, mass_frac=solid_comp)

            if isinstance(inputs['Inlet']['vol_flow'], float):
                vol_flow = inputs['Inlet']['vol_flow']
            else:
                vol_flow = inputs['Inlet']['vol_flow'][-1]

            if self.method == '1D-FVM':

                self.Outlet = SlurryStream(
                    vol_flow=vol_flow,
                    x_distrib=self.x_grid,
                    distrib=dp['distrib'][-1])

            elif self.method == 'moments':

                self.Outlet = SlurryStream(
                    vol_flow=vol_flow,
                    moments=dp['mu_n'][-1])

            self.get_heat_duty(time, states)  # TODO: allow for semi-batch

        else:
            liquid_out = copy.deepcopy(self.Liquid_1)
            solid_out = copy.deepcopy(self.Solid_1)

            self.Outlet = Slurry(vol=vol_slurry)

        # self.outputs = y_outputs
        self.Outlet.Phases = (liquid_out, solid_out)

    def get_heat_duty(self, time, states):
        q_heat = np.zeros((len(time), 3))

        if self.params_iter is None:
            merged_params = self.Kinetics.concat_params()[self.mask_params]
        else:
            merged_params = self.params_iter

        for ind, row in enumerate(states):
            row = row.copy()
            row[:self.num_distr] *= self.scale  # scale distribution
            q_heat[ind] = self.unit_model(time[ind], row, merged_params,
                                          enrgy_bce=True)

        # q_heat[:, 0] *= -1
        q_gen, q_ht, flow_term = q_heat.T  # TODO: controlled temperature

        self.heat_prof = q_heat
        self.heat_duty = np.array([0, trapezoidal_rule(time, q_ht)])
        self.duty_type = [0, -2]


class SemibatchCryst(MSMPR):
    """ Construct a Semi-batch Crystallizer object
    
    Parameters
    ----------
    target_comp : str, list of strings
        Name of the crystallizing compound(s) from .json file.
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computation
    method : str
        Choice of the numerical method. Options are: 'moments', '1D-FVM'
    scale : float
        Scaling factor by which crystal size distribution will be
        multiplied.
    controls : dict of dicts (funcs) (optional, default = None)
        Dictionary with keys representing the state (e.g.'Temp')
        which is controlled and the value indicating the function
        to use while computing the varible. Functions are of the form
        f(time) = state_value
    adiabatic : bool (optional, default =True)
        Boolean value indicating whether the heat transfer of
        the crystallization is considered.
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be
        reset before simulation
    basis : str (optional, default = 'mass_conc')
        Options : 'massfrac', 'massconc'
    state_events : lsit of dict(s)
        list of dictionaries, each one containing the specification of a
        state event
    """
    def __init__(self, target_comp, vol_tank=None, mask_params=None,
                 method='1D-FVM', scale=1, controls=None, adiabatic=False,
                 rad_zero=0, reset_states=False, h_conv=1000, vol_ht=None,
                 basis='mass_conc', jac_type=None, num_interp_points=3,
                 state_events=None, param_wrapper=None):

        super().__init__(target_comp, mask_params,
                         method, scale, vol_tank,
                         controls, adiabatic, rad_zero,
                         reset_states,
                         h_conv, vol_ht, basis,
                         jac_type, num_interp_points, state_events,
                         param_wrapper)

    def material_balances(self, time, params, u_inputs, rhos, mu_n,
                          distrib, mass_conc, temp, temp_ht, vol, phi_in):

        rho_susp, rho_in = rhos

        rho_liq, rho_sol = rho_susp
        rho_in_liq, _ = rho_in

        input_flow = u_inputs['Inlet']['vol_flow']
        input_flow = np.max([eps, input_flow])

        # TODO: generalize dictionary iteration ('Inlet', 'Liquid_1', ...)?
        input_distrib = u_inputs['Inlet']['distrib'] * self.scale
        input_conc = u_inputs['Liquid_1']['mass_conc']

        # print('time = %.2f, vol = %.2e, flowrate = %.2e' % (time, vol, input_flow))

        vol_solid = mu_n[3] * self.Solid_1.kv  # mu_3 is total, not by volume
        vol_slurry = vol + vol_solid

        self.Liquid_1.updatePhase(mass_conc=mass_conc)

        if self.method == 'moments':
            ddistr_dt, transf = self.method_of_moments(distrib, mass_conc, temp,
                                                       params, rho_sol,
                                                       vol=vol_slurry)

        elif self.method == '1D-FVM':
            ddistr_dt, transf = self.fvm_method(distrib, mu_n, mass_conc, temp,
                                                params, rho_sol,
                                                vol=vol_slurry)

        # ---------- Add flow terms
        # Distribution
        flow_distrib = input_flow * input_distrib

        ddistr_dt = ddistr_dt + flow_distrib

        # Liquid phase
        c_tank = mass_conc

        flow_term = phi_in[0]*input_flow * (
            input_conc - mass_conc * rho_in_liq/rho_liq)
        transf_term = transf * (self.kron_jtg - c_tank/rho_liq)

        dcomp_dt = 1/vol * (flow_term - transf_term)
        dvol_dt = (phi_in[0] * input_flow * rho_in_liq - transf) / rho_liq

        dliq_dt = np.append(dcomp_dt, dvol_dt)

        if self.basis == 'mass_frac':
            dcomp_dt *= 1 / rho_liq

        dmaterial_dt = np.concatenate((ddistr_dt, dliq_dt))

        return dmaterial_dt, transf

    def energy_balances(self, time, params, cryst_rate, u_inputs, rhos,
                        distrib, mass_conc, temp, temp_ht, vol, mu_n, h_in):

        rho_susp, rho_in = rhos

        # Input properties
        input_flow = u_inputs['Inlet']['vol_flow']
        input_flow = np.max([eps, input_flow])

        vol_solid = mu_n[3] * self.Solid_1.kv  # mu_3 is total, not by volume
        vol_total = vol + vol_solid

        phi = vol / vol_total
        phis = [phi, 1 - phi]
        dens_slurry = np.dot(rho_susp, phis)

        # Suspension properties
        capacitance = self.Slurry.getCp(temp, phis, rho_susp,
                                        times_vliq=True)
        h_sp = self.Slurry.getEnthalpy(temp, phis, rho_susp)

        # Renaming
        dh_cryst = -1.46e4  # J/kg
        # dh_cryst = -self.Liquid_1.delta_fus[self.target_ind] / \
        #     self.Liquid_1.mw[self.target_ind] * 1000  # J/kg

        # Terms
        dens_in_liq = rho_in[0]
        dmass_dt = input_flow * dens_in_liq

        accum_term = dmass_dt * h_sp/dens_slurry
        flow_term = input_flow * h_in

        source_term = dh_cryst * cryst_rate

        height_liq = vol / (np.pi/4 * self.diam_tank**2)
        area_ht = np.pi * self.diam_tank * height_liq + self.area_base  # m**2

        if self.adiabatic:
            ht_term = 0
        else:
            ht_term = self.u_ht*area_ht*(temp - temp_ht)

        # Balance inside the tank
        dtemp_dt = (flow_term - source_term - ht_term - accum_term) / \
            capacitance / vol

        # print(dtemp_dt)

        if temp_ht is not None:
            tht_in = self.temp_ht_in  # degC
            flow_ht = self.flow_ht
            cp_ht = 4180  # J/kg/K
            rho_ht = 1000
            vol_ht = self.vol_tank*0.14  # m**3

            dtht_dt = flow_ht / vol_ht * (tht_in - temp_ht) - \
                self.u_ht*area_ht*(temp_ht - temp) / rho_ht/vol_ht/cp_ht

            return dtemp_dt, dtht_dt

        else:
            return dtemp_dt
