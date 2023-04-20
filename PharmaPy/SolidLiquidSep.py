# -*- coding: utf-8 -*-
"""
Created on Wed May 13 02:03:18 2020

@author: huri
"""

import numpy as np
from PharmaPy.Commons import trapezoidal_rule, series_erfc
from PharmaPy.Phases import classify_phases
from PharmaPy.MixedPhases import Slurry, Cake
from PharmaPy.general_interpolation import define_initial_state

from PharmaPy.Commons import unpack_states, reorder_pde_outputs, eval_state_events, handle_events, unpack_discretized
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Results import DynamicResult
from PharmaPy.NameAnalysis import get_dict_states

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import copy

from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem
from assimulo.exception import TerminateSimulation

from scipy.special import erfc

eps = np.finfo(float).eps * 1.1
grav = 9.8  # m/s**2


def high_resolution_fvm(f, boundary_cond, limiter_type='Van Leer'):

    # Ghost cells -1, 0 and N + 1 (see LeVeque 2002, Chapter 9)
    f_extrap = 2*f[-1] - f[-2]
    f_aug = np.concatenate(([boundary_cond]*2, f, [f_extrap]))

    f_diff = np.diff(f_aug, axis=0)

    theta = (f_diff[:-1]) / (f_diff[1:] + eps)

    if limiter_type == 'Van Leer':
        limiter = (np.abs(theta) + theta) / (1 + np.abs(theta))
    else:  # TODO: include more limiters
        pass

    fluxes = f_aug[1:-1] + 0.5 * f_diff[1:] * limiter

    return fluxes


def upwind_fvm(f, boundary_cond):
    f_aug = np.concatenate(([boundary_cond], f))

    return f_aug


def get_alpha(solid_phase, porosity, sphericity, rho_sol, csd=None):
    # if csd is None:
    #     csd = solid_phase.distrib

    # x_grid = solid_phase.x_distrib

    # alpha_x = 180 * (1 - porosity) / \
    #     (porosity**3 * (x_grid*1e-6)**2 * rho_sol * sphericity**2)

    # numerator = trapezoidal_rule(x_grid, csd * alpha_x)
    # denominator = solid_phase.moments[0]

    # alpha = numerator / (denominator + eps)
    csd = solid_phase.distrib
    rho_sol = solid_phase.getDensity()
    x_grid = solid_phase.x_distrib * 1e-6

    kv = 0.524  # converting number based CSD to volume based:

    del_x_dist = np.diff(x_grid)
    node_x_dist = (x_grid[:-1] + x_grid[1:]) / 2
    node_CSD = (csd[:-1] + csd[1:]) / 2

    # Volume of crystals in each bin
    vol_cry = node_CSD * del_x_dist * (kv * node_x_dist**3)
    frac_vol_cry = vol_cry / (np.sum(vol_cry) + eps)

    csd = vol_cry

    # Calculate irreducible saturation in weighted csd (volume based)
    vol_frac = vol_cry/ np.sum(vol_cry)
    x_grid = node_x_dist
    alpha_x = 180 * (1 - porosity) / porosity**3 / x_grid**2 / rho_sol
    alpha = np.sum(alpha_x * vol_frac)

    return alpha


def get_sat_inf(x_vec, csd, deltaP, porosity, height, mu_zero, props):
    surf_tens, rho_liq = props

    kv = 0.524  # converting number based CSD to volume based:

    del_x_dist = np.diff(x_vec)
    node_x_dist = (x_vec[:-1] + x_vec[1:]) / 2
    node_CSD = (csd[:-1] + csd[1:]) / 2

    x_vec = node_x_dist
    if isinstance(surf_tens, float) or isinstance(rho_liq, float):
        capillary_number = porosity**3 * x_vec**2 * \
            (rho_liq*grav*height + deltaP) / (1 - porosity)**2 / height / surf_tens
    else:
        capillary_number = np.outer(
            porosity**3 * x_vec**2,
            (rho_liq*grav*height + deltaP)/(1 - porosity)**2 / height / surf_tens
            )
    # Volume of crystals in each bin
    vol_cry = node_CSD * del_x_dist * (kv * node_x_dist**3)
    frac_vol_cry = vol_cry / (np.sum(vol_cry) + eps)

    csd = vol_cry

    s_inf = 0.155 * (1 + 0.031*capillary_number**(-0.49))
    s_inf = np.where(s_inf > 1, 1, s_inf)

    # Calculate irreducible saturation in weighted csd (volume based)
    vol_frac = vol_cry/ np.sum(vol_cry)

    s_inf = np.sum(vol_frac *s_inf)

    return s_inf


class Carousel:

    def __init__(self, Phases=None, num_chambers=None, cycle_time=None):
        self._Phases = Phases
        self.uo_instances = {}

    def __CollectInstances(self):
        for key, value in self.__dict__.items():
            module = getattr(value, '__module__', None)
            if module == 'SolidLiqSep':
                self.uo_instances[key] = value


class DeliquoringStep:
    def __init__(self, num_nodes, diam_unit=0.01,
                 resist_medium=1e9):

        """

        Parameters
        ----------
        number_nodes : float
            Number of apatial discretization of the cake along the axial
            coordinate.
        diam_unit : float (optional, default=0.01)
            Diameter of the dryer's cross section [m]
        resist_medium : float (optional, default=1e9)
            Mesh resistance of filter in dryer. [m**-1]

        """

        self.num_nodes = num_nodes
        self.diam_unit = diam_unit
        self.area_cross = np.pi/4 * diam_unit**2
        self.resist_medium = resist_medium

        # Phases output
        self._Phases = None

        self.nomenclature()
        self.oper_mode = 'Batch'
        self.is_continuous = False

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if isinstance(phases, (list, tuple)):
            self._Phases = phases

            self.CakePhase = Cake(self.num_nodes)
            self.CakePhase.Phases = list(phases)
        elif phases.__class__.__name__ == 'Cake':
            self.CakePhase = phases
            self._Phases = phases.Phases
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')

        classify_phases(self)  # Enumerate phases: Liquid_1,..., Solid_1, ...

        self.cake_height = self.CakePhase.cake_vol / self.area_cross

        z_grid_red = np.linspace(0, 1, self.num_nodes + 1)
        dz = z_grid_red[1] - z_grid_red[0]

        z_centers = (z_grid_red[1:] + z_grid_red[:-1]) / 2

        self.z_centers = z_centers
        self.z_grid = z_grid_red
        self.delta_z = np.diff(z_grid_red)

        self.__original_phase__ = copy.deepcopy(self.Liquid_1.__dict__)

        self.name_species = self.Liquid_1.name_species

        self.states_di = {
            'saturation' :{# 'index': index_z,
                           'dim': 1,
                           'units': '', 'type': 'diff'},
            'mass_conc' : {'index': self.name_species,
                         'dim':len(self.name_species), 'units': 'kg/m**3', 'type':'diff'} # Order of the states matters
                }

        self.fstates_di = {
            'mean_saturation_value' : {# 'index': index_z,
                     'dim': 1, 'units': '',}}

        self.name_states = list(self.states_di.keys())
        self.dim_states = [a['dim'] for a in self.states_di.values()]

        self.nomenclature()

    def nomenclature(self):
        self.names_states_in = ['mass', 'temp',
                                'mass_frac', 'total_distrib']  #from filter states_out
        self.names_states_out = self.names_states_in

    def get_inputs(self, time):
        pass

    def unit_model(self, theta, states):
        states_reord = states.reshape(-1, self.Liquid_1.num_species + 1)
        sat_red = states_reord[:, 0]
        mass_conc = states_reord[:, 1:]

        model_eqns = self.material_balance(theta, sat_red, mass_conc)

        return model_eqns

    def material_balance(self, theta, sat_star, conc_star):

        """

        Calculate material balance for a non-dimensional version of the
        governing equations. Based on Wakeman

        """

        lambd = 5
        sat_star = np.where(sat_star<0, eps, sat_star) #Changing the negative element for numerical issue

        sat_aug = np.append(sat_star, sat_star[-1])
        p_liq = (self.p_gas - self.p_thresh*sat_aug**(-1/lambd))/self.p_thresh

        dpliq_dz = np.diff(p_liq)

        k_rl = sat_star**3.4

        q_liq = -k_rl * dpliq_dz  # Non-dimensional liquid flux

        sinf = self.sat_inf
        sat_fun = (1 - sinf) / (sat_star*(1 - sinf) + sinf)
        advection_vel = q_liq * sat_fun

        conc_bound = conc_star[0]  # dC/dt|_{z=0} = 0
        # conc_bound = np.zeros(conc_star.shape[1])  # F_{1 - 1/2} = 0
        # conc_bound = (np.zeros(conc_star.shape[1]) - self.conc_mean_init) / \
        #     (self.rho_j - self.conc_mean_init)

        flux_sat = upwind_fvm(q_liq, boundary_cond=0)
        # flux_conc = upwind_fvm((advection_vel * conc_star.T).T,
        #                         boundary_cond=conc_bound)
        flux_conc = upwind_fvm(conc_star, boundary_cond=conc_bound)

        numerical_fluxes = np.column_stack((flux_sat, flux_conc))

        dstates_dtheta = -np.diff(numerical_fluxes, axis=0).T / self.delta_z
        dstates_dtheta[1:] = dstates_dtheta[1:] * advection_vel

        return dstates_dtheta.T.ravel()

    def solve_unit(self, deltaP, runtime, p_atm=101325,
                   verbose=True):

        # Solid properties
        csd = self.Solid_1.distrib #* 1e6
        diam_i = self.Solid_1.x_distrib# * 1e-6  # um
        mom_zero = self.Solid_1.moments[0]
        alpha = self.CakePhase.alpha

        # Irreducible saturation
        epsilon = self.Solid_1.getPorosity()

        rho_liq_per_node = self.Liquid_1.getDensity()
        rho_liq = np.mean(rho_liq_per_node)
        self.visc_liq_per_node = self.Liquid_1.getViscosity()
        self.visc_liq = np.mean(self.visc_liq_per_node)
        surf_tens_per_node = self.Liquid_1.getSurfTension()
        surf_tens = np.mean(surf_tens_per_node)
        s_inf = get_sat_inf(diam_i, csd, deltaP, epsilon, self.cake_height,
                            mom_zero, (surf_tens, rho_liq))

        self.sat_inf = s_inf

        # Threshold pressure
        p_thresh = 4.6 * (1 - epsilon) * surf_tens / epsilon / diam_i
        p_thresh = trapezoidal_rule(diam_i, p_thresh*csd) / mom_zero

        self.p_thresh = p_thresh

        rho_s = self.Solid_1.getDensity()
        k_perm = 1 / alpha / rho_s / (1 - epsilon)

        # Gas pressure
        deltaP_media = deltaP*self.resist_medium / \
            (alpha*rho_s*self.cake_height*(1 - epsilon) + self.resist_medium)

        pgas_out = p_atm + deltaP - deltaP_media

        self.p_gas = np.linspace(pgas_out, p_atm, self.num_nodes + 1)

        # ---------- Initial states
        # Saturation # TODO: read from cake nodes
        sat_initial = np.ones(self.num_nodes) * self.CakePhase.saturation[0]
        sat_red_init = (sat_initial - s_inf) / (1 - s_inf)

        # Concentration
        self.rho_j = self.Liquid_1.getDensityPure()[0]
        # self.rho_j = np.ones_like(self.rho_j)
        conc_upstream = self.CakePhase.Liquid_1.mass_conc

        z_dim = self.z_centers * self.cake_height
        conc_init = define_initial_state(state=conc_upstream, z_after=z_dim,
                     z_before=self.CakePhase.z_external, indexed_state=True)

        # if conc_upstream.ndim == 1:     # also for saturation : Daniel
        #     z_dim = self.z_centers * self.cake_height
        #     conc_liq = self.Liquid_1.mass_conc  # Lets check how the mass conc is calculated: Daniel
        #     conc_init = np.tile(conc_liq, (self.num_nodes, 1))
        # elif conc_upstream.ndim == 2:
        #     z_dim = self.z_centers * self.cake_height
        #     interp = SplineInterpolation(self.CakePhase.z_external, conc_upstream)
        #     conc_init = interp.evalSpline(z_dim)

        self.conc_mean_init = np.zeros_like(conc_init)
        for i in range(len(self.Liquid_1.name_species)):
            self.conc_mean_init[:,i] = trapezoidal_rule(z_dim, conc_init[:,i]) / \
            self.cake_height

        conc_star_init = (conc_init - self.conc_mean_init) / \
            (self.rho_j - self.conc_mean_init)

        y_zero = np.column_stack((sat_red_init, conc_star_init)).ravel()

        model = Explicit_Problem(self.unit_model, y0=y_zero, t0=0)

        # Solve model
        model.name = 'Deliquoring PDE'
        sim = CVode(model)
        sim.linear_solver = 'SPGMR'

        self.theta_conv = k_perm*p_thresh / \
            self.visc_liq/self.cake_height**2/epsilon/(1 - s_inf)

        t_final = runtime * self.theta_conv
        
        if not verbose:
          sim.verbosity = 50
        
        if t_final < eps:
            t_final = 1
            
        theta, states = sim.simulate(t_final)

        self.rho_s = rho_s
        self.retrieve_results(theta, states)

        return theta, states

    def flatten_states(self):
        pass

    def retrieve_results(self, theta, states):
        num_species = self.Liquid_1.num_species

        time = theta/ self.theta_conv

        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        dp = {}

        dp_reduced= unpack_discretized(states, self.dim_states, self.name_states,
                                indexes=indexes)

        s_red = dp_reduced['saturation']

        # s_red = states[:, ::num_species + 1]
        satProf = s_red * (1 - self.sat_inf) + self.sat_inf

        conc_diff = self.rho_j - self.conc_mean_init

        concPerSpecies = {}
        mass_j = {}
        mass_bar_j = {}

        porosity = self.CakePhase.porosity

        for ind, name in enumerate(indexes['mass_conc']):
            conc_sp = dp_reduced['mass_conc'][name]

            conc_sp = conc_sp * conc_diff[:,ind] + self.conc_mean_init[:,ind]
            mass_sp = porosity * satProf * conc_sp
            massbar = porosity * satProf * conc_sp / \
                ((1 - porosity)*self.rho_s + porosity*satProf*self.rho_j[ind])

            concPerSpecies[name] = conc_sp
            mass_j[name] = mass_sp
            mass_bar_j[name] = massbar

        self.timeProf = time
        self.satProf = satProf

        dp['time'] = time
        dp['z'] = self.z_centers
        dp['saturation'] = self.satProf
        dp['mass_conc'] =  concPerSpecies

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        self.mean_sat = trapezoidal_rule(self.z_centers, s_red.T) * \
            (1 - self.sat_inf) + self.sat_inf

        # dp['mean_saturation_value'] = self.mean_sat


        concPerVolElement = {}
        concPerVolElement = dp_reduced['mass_conc']

        for name in indexes['mass_conc']:
            concPerVolElement[name] = concPerSpecies[name] * conc_diff[:, ind] \
                + self.conc_mean_init[:,ind]

        self.concPerSpecies = concPerSpecies
        self.massCompPerCakeUnitVolume = mass_j
        self.massjPerMassCake = mass_bar_j
        self.concPerVolElement = concPerVolElement

        last_state = {}
        for name in self.concPerSpecies.keys():
            last_state[name] = self.concPerSpecies[name][-1][-1]

        self.mass_conc= list(last_state.values())

        self.Liquid_1.updatePhase(mass_conc=self.mass_conc)


        liquid_out = copy.deepcopy(self.Liquid_1)
        solid_out = copy.deepcopy(self.Solid_1)

        self.Outlet = self.CakePhase
        self.CakePhase.saturation = self.satProf[-1]
        self.CakePhase.z_external = self.z_centers
        self.Outlet.Phases = (liquid_out, solid_out)
        self.outputs = states

    def plot_profiles(self, fig_size=None, mean_sat=True,
                      time=None, z_star=None, jump=20, pick_comp=None):
        """

        Parameters
        ----------
        fig_size : tuple (optional, default = None)
            Size of the figure to be populated.
        mean_sat : bool (optional, default = True)
            Boolean value indicating whether the
            averaged saturation value is plotted over the time.
        time : float (optional, default = None)
            Integer value indicating the time in which
            axial saturation value is calculated.
        z_star : float (optional, default = None)
            The axial coordinate of cake of interest to be calculated.
        jump : int (optional, default = 20)
            The number of sample to be skipped on the plot.
        pick_comp : list (optional, default = None)
            List contains the index of compounds of interest
            in the liquid phase to be calculated.

        """
        fig, axis = plt.subplots(2, 1, figsize=fig_size, sharex=True)

        if pick_comp is None:
            pick_comp = np.arange(len(self.Liquid_1.name_species))

        if mean_sat:
            axis[0].plot(self.timeProf, self.mean_sat)
            axis[0].set_xlabel('time (s)')
            axis[0].set_ylabel(r'$\bar{S}$')

            # axis[1].plot()

        elif time is None and z_star is None:
            # Saturation
            sat_plot = self.satProf.T[:, ::jump]
            num_lines = sat_plot.shape[1]
            axis[0].plot(self.z_centers, sat_plot)

            scale = np.linspace(0, 1, num_lines)
            colors = plt.cm.YlGnBu(scale)

            [line.set_color(color) for line, color in zip(axis[0].lines, colors)]

            axis[0].set_ylabel('$S$')

            # Concentration
            my_colors = plt.rcParams['axes.prop_cycle']()
            alphas = np.linspace(0.1, 1, num_lines)
            for ind in pick_comp:
                conc = self.concPerSpecies[ind]
                color = next(my_colors)
                axis[1].plot(self.z_centers, conc[::jump].T, **color)

                [line.set_alpha(alphas[idx])
                 for idx, line in enumerate(axis[1].lines[-num_lines:])]

                axis[1].lines[-1].set_label(self.Liquid_1.name_species[ind])

            axis[1].legend(loc='best')
            axis[1].set_ylabel('$C$ $(\mathregular{kg \ m^{-3}})$')
            axis[1].set_xlabel('$z/L$')

        elif time is not None:

            idx_time = np.argmin(abs(self.timeProf - time))
            profiles = [self.concPerSpecies[ind][idx_time]
                        for ind in range(self.Liquid_1.num_species)]

            axis[1].plot(self.z_centers, np.column_stack(profiles))

            axis[1].legend(self.Liquid_1.name_species, loc='best')

            axis[1].set_xlabel('$z/L$')
            axis[1].set_ylabel('$C_j$ ($\mathregular{kg \ m^{-3}}$)')
            axis[1].text(1, 1.04, '$t = %.1f$ s' % self.timeProf[idx_time],
                      ha='right', transform=axis[1].transAxes)

        elif z_star is not None:
            idx_z = np.argmin(abs(self.z_centers - z_star))
            profiles = self.concPerVolElement[idx_z]

            axis[1].plot(self.timeProf, profiles)
            axis[1].set_xlabel('time (s)')
            axis[1].set_ylabel('$C_j$ ($\mathregular{kg \ m^{-3}}$)')
            axis[1].text(1, 1.04, '$z^* = %.1f$' % self.z_centers[idx_z],
                         ha='right', transform=axis.transAxes)

        for ax in axis:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        return fig, axis

    def animate_unit(self, filename=None, title=None, pick_idx=None,
                     step_data=2, fps=5):
        if self.timeProf is None:
            raise RuntimeError

        if pick_idx is None:
            pick_idx = np.arange(self.Liquid_1.num_species)
            names = self.name_species
        else:
            pick_idx = pick_idx
            names = [self.name_species[ind] for ind in pick_idx]

        fig_anim, (ax_anim, ax_sat) = plt.subplots(2, 1, figsize=(4, 5))
        ax_anim.set_xlim(0, self.z_grid.max())
        fig_anim.suptitle(title)

        fig_anim.subplots_adjust(left=0, bottom=0, right=1, top=1,
                                 wspace=None, hspace=None)

        conc_min = np.vstack(self.concPerVolElement)[:, pick_idx].min()
        conc_max = np.vstack(self.concPerVolElement)[:, pick_idx].max()

        conc_diff = conc_max - conc_min

        ax_anim.set_ylim(conc_min - 0.03*conc_diff, conc_max + conc_diff*0.03)

        ax_anim.set_xlabel('$z/L$')
        ax_anim.set_ylabel('$C_j$ (mol/L)')

        sat = self.satProf
        sat_diff = sat.max() - sat.min()
        ax_sat.set_xlim(0, self.z_grid.max())
        ax_sat.set_ylim(sat.min() - sat_diff*0.03, sat.max() + sat_diff*0.03)

        ax_sat.set_xlabel('$z/L$')
        ax_sat.set_ylabel('$S$')

        def func_data(ind):
            conc_species = []
            for comp in pick_idx:
                conc_species.append(self.concPerSpecies[comp][ind])

            conc_species = np.column_stack(conc_species)
            return conc_species

        lines_conc = ax_anim.plot(self.z_centers, func_data(0))
        line_temp, = ax_sat.plot(self.z_centers, sat[0])

        time_tag = ax_anim.text(
            1, 1.04, '$time = {:.2f}$ s'.format(self.timeProf[0]),
            horizontalalignment='right',
            transform=ax_anim.transAxes)

        def func_anim(ind):
            f_vals = func_data(ind)
            for comp, line in enumerate(lines_conc):
                line.set_ydata(f_vals[:, comp])
                line.set_label(names[comp])

            line_temp.set_ydata(sat[ind])

            ax_anim.legend()
            fig_anim.tight_layout()

            time_tag.set_text('$t = {:.2f}$ s'.format(self.timeProf[ind]))

        frames = np.arange(0, len(self.timeProf), step_data)
        animation = FuncAnimation(fig_anim, func_anim, frames=frames,
                                  repeat=True)

        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'),
                              bitrate=-1)

        suff = '.mp4'

        animation.save(filename + suff, writer=writer)

        return animation, fig_anim, (ax_anim, ax_sat)


class Filter:
    def __init__(self, station_diam, alpha=None, resist_medium=1e9,
                 log_params=False):

        """

        Parameters
        ----------

        station_diam : float
            Diameter of the filter's cross section [m]
        alpha : float
            Specific cake resistance of filter cake. [m kg**-1]
        resist_medium : float (optional, default=1e9)
            Mesh resistance in filter. [m**-1]
        log_params : bool (optional, default = False)
            If true, alpha and resist_medium keyword should be
            provided in logarithmic scale.

        """
        self._Phases = None
        self.material_from_upstream = False

        self.r_medium = resist_medium
        self.station_diam = station_diam
        self.area_filt = station_diam**2 * np.pi/4  # [m^2]

        self.nomenclature()

        self.oper_mode = 'Batch'
        self.is_continuous = False

        self.alpha = alpha

        self.log_params = log_params

        self.elapsed_time = 0

        self.name_params = ['alpha', 'Rm']
        self.mask_params = [True, True]
        self.states_uo = ['mass_filtrate', 'mass_retained']

        self.deltaP = None
        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if isinstance(phases, (list, tuple)):
            self._Phases = phases

            self.SlurryPhase = Slurry()
            self.SlurryPhase.Phases = list(phases)
        elif phases.__class__.__name__ == 'Slurry':
            self.SlurryPhase = phases
            self._Phases = phases.Phases
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')
        classify_phases(self)  # Enumerate phases: Liquid_1,..., Solid_1, ...

        epsilon = self.Solid_1.getPorosity(diam_filter=self.station_diam)
        dens_sol = self.Solid_1.getDensity()
        if self.alpha is None:
            self.alpha = get_alpha(self.Solid_1, sphericity=1,
                                    porosity=epsilon,
                                    rho_sol=dens_sol)

        self.params = (self.alpha, self.r_medium)
        if self.log_params:
            self.params = np.log(self.params)

        # self.__original_phase__ = copy.deepcopy(self.Liquid_1.__dict__)
        # self.__original_phase__ = copy.deepcopy(self.Liquid_1)

        self.states_di = {
            'mass_filtrate': {'dim': 1, 'units': 'kg', 'type': 'diff'},
            'mass_liquid': {'dim': 1, 'units': 'kg', 'type': 'diff'},
            }

        self.fstates_di = {
            'mass_cake_dry': {'dim': 1, 'units': 'kg', 'type': 'alg'},
            'mass_cake_wet': {'dim': 1, 'units': 'kg', 'type': 'alg'}

            }

        self.name_states = list(self.states_di.keys())
        self.dim_states = [a['dim'] for a in self.states_di.values()]

        self.nomenclature()

    def nomenclature(self):
        self.names_states_in = ['mass', 'temp', 'mass_frac', 'total_distrib']
        self.names_states_out = self.names_states_in

    def get_inputs(self):
        self.Inlet

    def unit_model(self, time, states, sw):
        if self.oper_mode == 'Batch':
            mass_liquid = states
            material_balance = self.material_balance(time, mass_liquid,
                                                     self.deltaP)

        # print(material_balance)
        # print(time)
        # print(mass_liquid)

        return material_balance

    def material_balance(self, time, masses, dP):
        mass_filtr, mass_up = masses
        rho, visc, tens, rho_s = self.physical_props
        c_solids = self.c_solids

        alpha, resist = self.params

        cake_term = alpha * c_solids * mass_filtr / (self.area_filt * rho)**2
        filt_term = resist / (self.area_filt * rho)

        deriv = dP / visc / (cake_term + filt_term)

        dmass_dt = [deriv, -deriv]

        return dmass_dt

    def __state_event(self, time, states, switch):
        events = []
        if switch[0]:
            event_mass = self.mass_crit - states[0]
            events.append(event_mass)

        return np.array(events)

    def __handle_event(self, solver, event_info):
        state_event = event_info[0]

        if state_event:
            raise TerminateSimulation

    def reset(self):
        self.elapsed_time = 0
        if self.log_params:
            self.params = np.log(self.params)

    def solve_unit(self, runtime=None, time_grid=None, deltaP=1e5,
                   slurry_div=1, verbose=True, model_params=None,
                   sundials_opts=None):

        # Filtration parameters (constant)
        self.deltaP = deltaP

        epsilon = self.Solid_1.getPorosity(diam_filter=self.station_diam)
        dens_sol = self.Solid_1.getDensity()

        if model_params is not None:
            self.params = model_params

        if callable(self.alpha):
            self.alpha = self.alpha(deltaP)

        if self.log_params:
            self.params = np.exp(self.params)

        solid_conc = self.SlurryPhase.getSolidsConcentr()
        solid_conc = max(0, solid_conc)

        # Initial state
        vol_slurry = self.SlurryPhase.vol / slurry_div
        frac_liq = self.SlurryPhase.getFractions()[0]

        vol_liq_cake = vol_slurry * solid_conc/dens_sol * epsilon/(1 - epsilon)
        vol_liq_slur = vol_slurry * frac_liq

        vol_filtrate = vol_liq_slur - vol_liq_cake

        self.c_solids = vol_slurry * solid_conc / vol_filtrate

        # Physical properties
        dens_liq = self.Liquid_1.getDensity()
        visc_liq = self.Liquid_1.getViscosityMix()
        surf_liq = self.Liquid_1.getSurfTension()

        self.physical_props = [dens_liq, visc_liq, surf_liq, dens_sol]
        self.mass_crit = vol_filtrate * dens_liq

        # Initial values
        mass_filtr_init = 0
        mass_up_init = vol_liq_slur * dens_liq

        mass_init = [mass_filtr_init, mass_up_init]

        # Solve ODE
        problem = Explicit_Problem(self.unit_model, y0=mass_init, t0=0,
                                   sw0=[True])

        # State event
        if model_params is None:
            problem.state_events = self.__state_event
            problem.handle_event = self.__handle_event

        solver = CVode(problem)

        if not verbose:
            solver.verbosity = 50

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        if time_grid is not None:
            final_time = time_grid[-1] + self.elapsed_time
            time_grid += self.elapsed_time

        elif runtime is None:
            final_time = 1e10

        else:
            final_time = runtime + self.elapsed_time

        time, states = solver.simulate(final_time, ncp_list=time_grid)

        # Additional model equations
        self.time_filt = visc_liq/self.deltaP * (
            self.alpha*self.c_solids/2 * (vol_filtrate/self.area_filt)**2 +
            self.r_medium * (vol_filtrate/self.area_filt))
        # self.time_filt = visc_liq * self.alpha * vol_filtrate * solid_conc * vol_slurry\
        #     / (2 * self.area_filt**2 * self.deltaP)\
        #     + visc_liq * self.r_medium * vol_filtrate/ (self.area_filt * self.deltaP)
        self.retrieve_results(time, states, dens_liq, dens_sol, epsilon)

        return time, states

    def retrieve_results(self, time, states, dens_liq, dens_sol, epsilon):
        self.timeProf = np.array(time)
        self.massProf = states

        dp = unpack_states(states, self.dim_states, self.name_states)
        dp['time'] = np.asarray(time)

        cake_dry = self.c_solids * states[:, 0] / dens_liq
        cake_wet = cake_dry * (1 + epsilon/(1 - epsilon) * dens_liq/dens_sol)

        dp['mass_cake_dry'] = cake_dry
        dp['mass_cake_wet'] = cake_wet

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        solid_cake = copy.deepcopy(self.Solid_1)
        solid_cake.updatePhase(mass=cake_dry[-1], distrib=self.Solid_1.distrib)

        liquid_cake = copy.deepcopy(self.Liquid_1)
        liquid_cake.updatePhase(mass=self.massProf[-1, 1])  # TODO: the other outlet

        self.Outlet = Cake()
        self.Outlet.Phases = (liquid_cake, solid_cake)

        self.outputs = np.concatenate(([states[-1, 1], self.Liquid_1.temp],
                                       self.Liquid_1.mass_frac,
                                       self.Solid_1.distrib))

        self.outputs = np.atleast_2d(self.outputs)

        self.elapsed_time += time[-1]

    def flatten_states(self):
        pass

    def paramest_wrapper(self, params, time_vals, modify_phase=None,
                         modify_controls=None):

        self.reset()

        deltaP = self.deltaP
        time, states = self.solve_unit(time_grid=time_vals, deltaP=deltaP,
                                       model_params=params, verbose=False)

        return states[:, 0]

    def plot_profiles(self, time_div=1, black_white=False, **fig_kwargs):
        """

        Parameters
        ----------

        fig_size : tuple (optional, default = None)
            Size of the figure to be populated.
        time_div : float (optional, default = 1)
            The float value used to scale time value in
            plotting by dividing the simulated time.
        black_white : bool (optional, default = False)
            Boolean value indicating whether the figure
            is presented in black and white style.

        """

        mass_filtr, mass_up = self.massProf.T
        time_plot = self.timeProf / time_div

        fig, ax = plt.subplots(1, 2, **fig_kwargs, sharex=True)

        # Liquid mass
        if black_white:
            ax[0].plot(time_plot, mass_filtr, time_plot, mass_up, '--',
                       color='k')
            ax[1].plot(time_plot, self.cake_dry, time_plot, self.cake_wet,
                       '--', color='k')
        else:
            ax[0].plot(time_plot, mass_filtr, time_plot, mass_up, '--')
            ax[1].plot(time_plot, self.result.mass_cake_dry,
                       time_plot, self.result.mass_cake_wet,
                       '--')

        ax[0].set_ylabel('mass liquid (kg)')
        ax[0].legend(('filtrate', 'hold-up'))

        # Cake
        ax[1].set_ylabel('mass cake (kg)')
        ax[1].legend(('dry cake', 'wet cake'), loc='best')

        for axis in ax:
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)

            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        if time_div == 1:
            ax[1].set_xlabel('time (s)')

        return fig, ax


class DisplacementWashing:
    def __init__(self, solvent_idx, num_nodes, diam_unit=None,
                 resist_medium=1e9, k_ads=0):

        """

        Parameters
        ----------

        solvent_idx : int
            Integer value indicating the index of the compounds
            used as solvent. Index correpond to the coumpounds
            order in the physical properties .json file.
        number_nodes : float
            Number of apatial discretization of the cake along the axial
            coordinate.
        diam_unit : float (optional, default=0.01)
            Diameter of the dryer's cross section [m]
        resist_medium : float (optional, default=2.22e9)
            Mesh resistance of filter in dryer. [m**-1]
        k_ads : float (optional, default = 0)
            Equilibirum coefficient between main flow concentration
            and overall partical concentration.Used to calculate adsoprtion
            factor.The default value '0' means no adsorption occurs on the solid phase.
            (Lapidus and Amundson, 1952)

        """
        self.max_exp = np.log(np.finfo('d').max)
        self.satur = 1
        self.num_nodes = num_nodes

        self.solvent_idx = solvent_idx
        self.k_ads = k_ads
        self.cross_area = np.pi / 4 * diam_unit**2
        self.diam_unit = diam_unit

        self.resist_medium = resist_medium

        self._Phases = None
        self._Inlet = None

        self.oper_mode = 'Batch'
        self.is_continuous = False

        self.nomenclature()

        self.outputs = None

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if isinstance(phases, (list, tuple)):
            self._Phases = phases

            self.CakePhase = Cake()
            self.CakePhase.Phases = list(phases)
        elif phases.__class__.__name__ == 'Cake':
            self.CakePhase = phases
            self._Phases = phases.Phases
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')

        classify_phases(self)  # Enumerate phases: Liquid_1,..., Solid_1, ...

        self.__original_phase__ = copy.deepcopy(self.Liquid_1.__dict__)

        self.name_species = self.Liquid_1.name_species

        self.states_di = {
            'mass_conc' : {'index': self.name_species,
                         'dim':len(self.name_species), 'units': 'kg/m**3', 'type':'algebraic'}} # Order of the states matters

        self.fstates_di = {
            'mean_mass_conc' : {'index': self.name_species,
                     'dim': len(self.name_species), 'units': 'kg/m**3', 'type': 'alg'},
            'washing_time' : {#'index': index_z,
                              'dim':1, 'units': 's', 'type': 'alg'}}

        self.name_states = list(self.states_di.keys())
        self.dim_states = [a['dim'] for a in self.states_di.values()]

        self.nomenclature()
    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):
        self._Inlet = inlet

    def get_diffusivity(self, vel, diff_pure):
        distr = self.Solid_1.distrib
        size = self.Solid_1.x_distrib

        re_sc = vel * np.dot(distr, size)/diff_pure
        diff_ratio = np.ones_like(re_sc) * 1/np.sqrt(2)

        for i in range(len(re_sc)):

            if re_sc[i] > 1:
                diff_ratio[i] += 55.5 * re_sc[i]**0.96

        diff_eff = diff_ratio * diff_pure

        return diff_eff

    def nomenclature(self):
        self.names_states_in = ['mass_conc', 'temp',
                                'mass_frac', 'total_distrib']  #from filter states_out
        self.names_states_out = self.names_states_in

    def get_inputs(self, time):
        pass

    def material_balance(self, z_pos, time, vel, diff, lambd):

        arg_one = 0.5 * np.sqrt(lambd*vel**2*np.einsum('i,j->ij', time,
                                                       1/diff))
        arg_two = 0.5 * np.sqrt(1/lambd) * np.einsum(
            'i,j,k->ijk', z_pos, 1/np.sqrt(time), 1/np.sqrt(diff))
        arg_exp = vel * np.einsum('i,j->ij', z_pos, 1/diff)

        first = erfc(arg_one - arg_two)
        second = np.exp(arg_exp) * erfc(arg_one + arg_two).transpose(1, 0, 2)

        conc_star = 0.5 * (first - second.transpose(1, 0, 2))

        return conc_star

    def material_bce(self, z_pos, wash_ratio, vel, l_total, diff, lambd):
        # z_pos = np.atleast_1d(z_pos)

        root = np.sqrt(vel*l_total / diff)

        z_adim = z_pos / l_total
        lambd_wash = lambd * wash_ratio

        arg_one = (z_adim - lambd_wash)/2/np.sqrt(lambd_wash)
        arg_two = (z_adim + lambd_wash)/2/np.sqrt(lambd_wash)
        exp_term = np.exp(vel * np.outer(z_pos, 1/diff))

        conc_star = 1 - 0.5 * (
            erfc(np.outer(arg_one, root)) +
            exp_term * erfc(np.outer(arg_two, root))
            )

        return conc_star

    def solve_unit(self, deltaP, wash_ratio=1, time_vals=None,
                   dynamic=True, verbose=True):

        # ---------- Physical properties
        # Liquid
        visc_liq_per_node = self.Liquid_1.getViscosity()
        visc_liq = np.mean(visc_liq_per_node)
        diff_pure = self.Liquid_1.getDiffusivityPure(wrt=self.solvent_idx)
        epsilon = self.Solid_1.getPorosity(diam_filter=self.diam_unit)
        lambd_ads = 1 / (1 - self.k_ads + self.k_ads/epsilon)

        c_zero = np.array(self.CakePhase.Liquid_1.mass_conc)

        # Solid
        epsilon = self.Solid_1.getPorosity(diam_filter=self.diam_unit)
        dens_sol = self.Solid_1.getDensity()
        alpha = self.CakePhase.alpha

        # Cake
        cake_height = self.CakePhase.cake_vol / self.cross_area  # m
        vel_liq = deltaP / visc_liq / (alpha * dens_sol * cake_height *
                                       (1 - epsilon) + self.resist_medium)
        diff = self.get_diffusivity(vel_liq, diff_pure)

        z_vals = np.linspace(0, cake_height, self.num_nodes)
        c_zero = define_initial_state(state=c_zero, z_after=z_vals,
                     z_before=self.CakePhase.z_external, indexed_state=True)
        # if c_zero.ndim == 1:     # also for saturation : Daniel
        #     c_zero = np.tile(c_zero, (self.num_nodes, 1))
        # elif c_zero.ndim == 2:
        #     interp = SplineInterpolation(self.CakePhase.z_external, c_zero)
        #     c_zero = interp.evalSpline(z_vals)

        # self.conc_mean_init = np.zeros_like(conc_init)
        # for i in range(len(self.Liquid_1.name_species)):
        #     self.conc_mean_init[:,i] = trapezoidal_rule(z_dim, conc_init[:,i]) / \
        #     self.cake_height
        # c_zero = self.Liquid_1.mass_conc

        c_inlet = np.zeros(self.Liquid_1.num_species)
        c_inlet[self.solvent_idx] = self.Liquid_1.rho_liq[self.solvent_idx]

        # ---------- Solve
        time_total = wash_ratio * cake_height / vel_liq

        if time_vals is None:
            time_vals = np.linspace(eps, time_total)
        elif time_vals[-1] > time_total:
            raise RuntimeError('Final time higher than total time')

        self.num_z = len(z_vals)
        self.num_t = len(time_vals)

        if dynamic:
            conc_adim = self.material_balance(z_vals, time_vals, vel_liq, diff,
                                              lambd_ads)

            conc_star = conc_adim[:, -1]
        else:
            conc_adim = self.material_bce(z_vals, wash_ratio, vel_liq,
                                          cake_height, diff, lambd_ads)

            conc_star = conc_adim

        conc = conc_star * (c_zero - c_inlet) + c_inlet
        conc_all = np.zeros_like(conc_adim)

        for i in range(self.num_t):
            conc_all[:,i] = conc_adim[:,i] * (c_zero - c_inlet) + c_inlet
        # conc_all = conc_adim * (c_zero - c_inlet) + c_inlet

        # Average final concentration and material balance
        integral = trapezoidal_rule(z_vals, conc_star)
        c_cake = (c_zero - c_inlet) / cake_height * integral + c_inlet

        sat_zero = self.satur

        c_effl = (epsilon/wash_ratio * (sat_zero * c_zero - c_cake) + c_inlet) / \
            (1 + epsilon/wash_ratio * (sat_zero - 1))

        self.retrieve_results(z_vals, time_vals, conc_all)
        self.cake_height = cake_height
        self.conc_cake = c_cake
        self.conc_effl = c_effl

        return conc, conc_star, c_cake, c_effl

    def retrieve_results(self, z_coord, time_coord, conc):
        num_species = self.Liquid_1.num_species

        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        conc_T = np.transpose(conc, (1, 0, 2))

        dp= unpack_discretized(conc_T, self.dim_states, self.name_states,
                               indexes=indexes)

        self.zProf = z_coord
        self.timeProf = time_coord
        self.concProf = conc

        dp['time'] = np.asarray(self.timeProf)
        dp['z'] = self.zProf

        self.result = DynamicResult(self.states_di, self.fstates_di, **dp)

        if conc.ndim == 3:
            concPerVolElem = []
            for ind in range(self.num_z):
                concPerVolElem.append(conc[ind])

            concPerSpecies = []
            for ind in range(self.num_t):
                concPerSpecies.append(conc[:, ind])

            self.concPerVolElem = concPerVolElem
            self.concPerSpecies = concPerSpecies

        self.CakePhase.mass_concentr = self.concPerSpecies[-1]  # TODO
        self.CakePhase.z_external = self.zProf

        last_state = self.concPerSpecies[-1]
        self.Liquid_1.updatePhase(mass_conc=last_state)

        liquid_out = copy.deepcopy(self.Liquid_1)
        solid_out = copy.deepcopy(self.Solid_1)

        self.Outlet = self.CakePhase
        self.Outlet.Phases = (liquid_out, solid_out)
        if conc.ndim == 3:
            self.outputs = concPerVolElem[-1]

    def flatten_states(self):
        pass

    def plot_profiles(self, fig_size=None, z_val=None, time=None,
                      pick_idx=None):

        """

        Parameters
        ----------

        fig_size : tuple (optional, default = None)
            Size of the figure to be populated.
        z_val : int (optional, default=None)
            Integer value indicating the axial position of cake coordniate
            at which calculated washing outputs to be plotted.
        time : int (optional, default=None)
            Integer value indicating the time on which calculated washing
            outputs to be plotted.
        pick_idx : tuple of lists (optional, default=None)
            List of index of components to include in plotting.
            Length of tuple is 2. Index 0 and 1 corresponds to index of compounds
            in liquid phase and gas phase respectively.
            If None, all the existing components are plotted

        """

        fig, ax = plt.subplots(figsize=fig_size)

        if pick_idx is None:
            pick_idx = np.arange(self.Liquid_1.num_species)
        else:
            pick_idx = list(pick_idx)

        if self.concProf.ndim == 2:
            t_idx = np.argmin(abs(self.timeProf - time))
            ax.plot(washer.z_grid, self.concProf)
            ax.set_xlabel('$z$ (m)')
            ax.text(1, 1.04, '$t = %.0f$ s' % self.timeProf[t_idx],
                    ha='right', transform=ax.transAxes)
        else:
            if z_val is not None:
                z_idx = np.argmin(abs(self.zProf - z_val))

                ax.plot(self.t_grid, self.concPerVolElem[z_idx])
                ax.set_xlabel('time (s)')
                ax.text(1, 1.04, '$z = %.2e$ m' % self.zProf[z_idx],
                        ha='right', transform=ax.transAxes)

            if time is not None:
                t_idx = np.argmin(abs(self.timeProf - time))
                conc_plot = self.concPerSpecies[t_idx][:, pick_idx]

                ax.plot(self.zProf, conc_plot)
                ax.set_xlabel('$z$ (m)')
                ax.text(1, 1.04, '$t = %.0f$ s' % self.timeProf[t_idx],
                        ha='right', transform=ax.transAxes)

        ax.set_ylabel('$C_i$ $(\mathregular{kg \ m^{-3}})$')

        for ind in pick_idx:
            ax.legend(self.Liquid_1.name_species[ind], loc='best')

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        return fig, ax


if __name__ == '__main__':
    plt.style.use(
        '../../../publications/2021_CACE/source/cace_palatino.mplstyle')

    case = 1
    washer = DisplacementWashing()
    wratio = 0.2

    if case == 1:

        concentr, cstar, conc_cake, conc_receiver = washer.solve_unit(
            1.01325e5, wash_ratio=wratio, dynamic=False)

        figw, axw = washer.plot_profiles(fig_size=(4, 2.8), time=100)
                                         # z_val=0.05)

        axw.text(0, 1.1, '$C_{0, A} = C_{0, B} = 2$,   '
                 '$W = %.1f$, $C_{in, A} = C_{in, B} = 0$' % wratio,
                 transform=axw.transAxes)

    elif case == 2:  # Based on Lapidus and
        length = 0.05  # m
        vel = 0.001  # m/s
        diff = np.array([1e-5, 1e-5]) * 1e-4**0  # m**2/s
        time_total = wratio*length / vel

        length = 50
        vel = 400
        diff = vel / np.array([1, 2, 10, 100])  # m**2/s
        time_total = 2400 / vel
        times = np.linspace(1e-6, time_total, 100)

        z_vals = np.linspace(0, length, 50)

        cstar = washer.material_balance(z_vals, times, vel, diff)
        c_zero = 2
        c_in = 0

        conc_real = cstar * (c_zero - c_in) + c_in

        figw, axw = plt.subplots(1, 2, figsize=(6, 2))

        volume = vel * times * 1
        axw[0].plot(volume, cstar[-1, :, :-1])

        axw[0].set_xlabel(r'$t$')
        axw[0].set_ylabel('$\dfrac{C - C_0}{C_{in} - C_0}$')

        axw[1].plot(z_vals, conc_real[:, -1, :-1])
        axw[1].set_xlabel('$z$')
        axw[1].set_ylabel('$\dfrac{C - C_0}{C_{in} - C_0}$')
