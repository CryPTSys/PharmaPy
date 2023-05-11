#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:10:19 2019

@author: dcasasor
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import pandas as pd
from itertools import combinations
import time


class ParallelProblem:
    def __init__(self, instance):
        self.instance = instance

    def run_fit(self, sample):
        self.instance.y_fit = sample

        # Optimize
        params, _, _ = self.instance.optimize_fn(
            method=self.instance.opt_method,
            verbose=False, store_iter=False,
            optim_options=self.instance.optim_options)

        return params


class StatisticsClass:

    def __init__(self, estimation_instance, alpha=0.95):

        # Values from the instance
        inst = estimation_instance
        jac = inst.info_opt['jac']

        self.a_matrix = jac.dot(jac.T)
        self.a_inv = np.linalg.inv(self.a_matrix)
        self.jac = jac

        self.params = inst.params_convg.copy()
        self.params_unactive = inst.param_seed[inst.map_fixed]

        self.y_nominal = inst.y_model

        self.residuals = inst.resid_runs

        self.param_names = inst.name_params

        # Calculations
        self.dof = inst.num_data_total - inst.num_params

        residuals = inst.info_opt['fun'].copy()
        resid_squared = np.dot(residuals.T, residuals)  # SSE
        self.var_fun = resid_squared / self.dof

        self.num_par = len(self.params)

        self.xgrid = inst.x_data

        self.inst = inst

        self.alpha = alpha

        # --------------- Pair-wise combinations
        # Indexes and names
        self.combs = tuple(combinations(range(self.num_par), 2))
        self.name_pairs = []
        for comb in self.combs:
            names = [self.param_names[ind] for ind in comb]
            name_pair = '/'.join(names)
            self.name_pairs.append(name_pair)

        # --------------- Output vals
        self.confid_intervals = None
        self.sigma_t = None
        self.boot_params = None

    def get_intervals(self, covar_params=None, verbose=True, set_self=True):

        if covar_params is None:
            covar_params = self.inst.covar_params

        sigma_params = np.sqrt(np.diag(covar_params))

        t_stat = stats.t.ppf((1 + self.alpha)/2, self.dof)

        delta_par = sigma_params * t_stat

        intervals = np.column_stack((self.params - delta_par,
                                     self.params + delta_par))

        confidence_int = dict(zip(self.param_names, intervals))

        perc_deviation = np.abs(delta_par / self.params * 100)

        if verbose:
            alperc = self.alpha * 100
            print('')
            print('{:<55}'.format('-'*100))
            print("{:<12} {:^15} {:^15} {:^15} {:^15} {:^15}".format(
                    'param', 'lb', 'mean', 'ub',
                    "%i%% CI (+/-)" % alperc, "%i%% CI (+/- %%)" % alperc))

            print('{:<55}'.format('-'*100))

            for ind, (key, val) in enumerate(confidence_int.items()):
                low, high = val
                print("{:<12} {:^15.3e} {:^15.3e} {:^15.3e} {:^15.3e} {:^15.5f}".format(
                        key, low, self.params[ind], high, delta_par[ind], perc_deviation[ind]))

            print('{:<55}'.format('-'*100))

        if set_self:
            self.confid_intervals = {'intervals': confidence_int,
                                     'delta_param': delta_par,
                                     'perc_deviation': perc_deviation}

            self.covar_params = covar_params
            self.intervals_array = intervals

        return confidence_int

    def confidence_regions(self, param_means, alphas=None):
        """
        Generate pairwise confidence ellipses. For three or more parameters,
        the 2D projections are calculated as described in 'Nonlinear
        Regression' by Seber and Wilde (1989).

        Parameters
        ----------
        alphas : tuple or list, optional
            confidence levels to draw ellipses. If None, the alpha value
            specified when creating the instance is used.
            The default is None.

        Returns
        -------
        info_dict : dict of dicts
            dictionary with keys as the paramter pair names, e.g.
            'theta_1/theta_2'. Each dictionary value is in turn a dictionary
            with the ellipse and rectangle patches and their respective
            dimensions (axes lenghts or side lengths)

        """

        info_dict = {}
        indexes = set(range(self.num_par))
        ident = np.eye(self.jac.shape[1])
        for comb, name in zip(self.combs, self.name_pairs):
            comb = list(comb)

            param_pair = param_means[comb].values
            diff = list(indexes.difference(comb))

            if len(diff) == 0:
                a_projec = self.a_matrix
                a_inv = self.a_inv
            else:
                f_1 = self.jac[diff].T
                f_2 = self.jac[comb].T

                projection = f_1.dot(np.linalg.inv(f_1.T.dot(f_1))).dot(f_1.T)
                a_projec = f_2.T.dot(ident - projection).dot(f_2)
                a_inv = np.linalg.inv(a_projec)

            el, rec, axlengs, recdims = self.__confidence_pair(
                a_projec, a_inv, param_pair, alphas=alphas)

            info_dict[name] = {'ellipses': el, 'rectangles': rec,
                               'ellipse_axes': axlengs,
                               'rectangle_dims': recdims}

        return info_dict

    def __confidence_pair(self, a_projected, a_inverse, param_pair,
                          alphas=None):

        # Select eigenvals, eigenvecs and params
        eig_vals, eig_vecs = np.linalg.eig(a_projected)

        # Statistic
        if alphas is None:
            alphas = [self.alpha]

        f_stat = stats.f.ppf(alphas, self.num_par, self.dof)  # TODO: I think self.num_par should be just 2
        contours = f_stat * self.num_par * self.var_fun

        # Order
        orders = np.argsort(eig_vals)[::-1]

        eig_vals = eig_vals[orders]
        eig_vecs = eig_vecs[:, orders]

        # Get rotation angle (angle of the largest eigenvector)
        angle = np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0])
        angle = np.degrees(angle)

        ellipses = []
        axes_len = []

        rectangles = []
        width_height = []
        for contour in contours:
            axes = np.sqrt(contour / eig_vals)  # half width
            axes *= 2

            # Draw ellipse
            ellipse = Ellipse(param_pair, *axes, angle=angle,
                              alpha=0.2, edgecolor='k', linestyle='--',
                              zorder=5)

            # Draw rectangle
            side_len = np.sqrt(contour * np.diag(a_inverse))
            width = side_len[0] * 2
            height = side_len[1] * 2

            rectangle = Rectangle(param_pair - side_len, width, height,
                                  fill=False, zorder=5, alpha=0.4)

            # Store
            ellipses.append(ellipse)
            axes_len.append(axes)

            rectangles.append(rectangle)
            width_height.append(np.array([width, height]))

        return ellipses, rectangles, axes_len, width_height

    def plot_pairwise(self, boots=None, alphas=None, bounding_box=False,
                      interval_bounds=True, fig_size=(5, 5)):

        boot_params = boots

        if boot_params is None:
            boot_params = self.boot_params
            confid_interv = self.intervals_array

        else:
            covar_boots = np.cov(boot_params.T)
            confid_interv = self.get_intervals(covar_boots, verbose=False,
                                               set_self=False)

            confid_interv = np.vstack(list(confid_interv.values()))

        if boot_params is None:
            raise RuntimeError(
                "No data to plot. Run 'bootstrap_params' method first ")

        params = pd.DataFrame(boot_params)
        param_means = params.mean(axis=0)

        axes = pd.plotting.scatter_matrix(params, figsize=fig_size,
                                          hist_kwds={'bins': 50}, zorder=10,
                                          density_kwds={'s': 10})

        for ind in range(self.num_par):
            axes[ind, 0].set_ylabel(self.inst.name_params_plot[ind])
            axes[self.num_par - 1, ind].set_xlabel(
                self.inst.name_params_plot[ind])

        regions = self.confidence_regions(param_means, alphas=alphas)
        for ind, vals in enumerate(regions.values()):
            ovals = vals['ellipses']
            bounding = vals['rectangles']
            idx = self.combs[ind][::-1]  # lower triangular quadrant index

            for region, box in zip(ovals, bounding):

                axes[idx].add_patch(region)

                if interval_bounds:
                    # Show confidence intervals
                    intervals = confid_interv[idx[::-1], :]
                    for row in intervals.T:
                        axes[idx].axvline(row[0], ls='--', zorder=0)
                        axes[idx].axhline(row[1], ls='--', zorder=0)

                # axes[idx].autoscale()

                if bounding_box:
                    axes[idx].add_patch(box)

            param_pair = param_means[list(idx)[::-1]].values
            axes[idx].scatter(*param_pair, marker='+', color='k')

            width_height = vals['rectangle_dims']
            rect_array = np.array(width_height)

            max_dims = rect_array.max(axis=0) * 1.2

            fac = 0.5

            axes[idx].set_xlim(param_pair[0] - fac*max_dims[0],
                               param_pair[0] + fac*max_dims[0])

            axes[idx].set_ylim(param_pair[1] - fac*max_dims[1],
                               param_pair[1] + fac*max_dims[1])

        # Hide upper triangular
        mask = np.triu_indices_from(axes, 1)

        for row, col in zip(*mask):
            axes[row, col].set_visible(False)

        axes[0, 0].yaxis.set_ticks([])

        return axes

    def bootstrap_confid(self, num_samples, plot=True):
        std_samples = np.random.randn(num_samples, self.num_par)

        stdev_params = np.array(list(self.sigma_t.values()))
        samples = std_samples * stdev_params**2 + self.params

        # Concentration profiles with the optimum params
        conc_nominal = self.inst.conc_profile.copy()

        conc_bootstrap = []
        for sample in samples:
            self.inst.assign_params(sample)
            self.inst.solve_model(self.c_init, time_eval=self.xgrid,
                                  print_summary=False)

            conc_bootstrap.append(self.inst.conc_profile)

        if plot:
            fig, axis = plt.subplots()
            axis.plot(self.xgrid, conc_nominal[:, self.inst.measured_ind])

            lines = axis.lines
            colors = [line.get_color() for line in lines]

            for array in conc_bootstrap:
                for ind in range(self.inst.num_measured):
                    axis.plot(self.xgrid, array[:, ind], color=colors[ind],
                              alpha=0.02)

        return samples, conc_bootstrap, fig, axis

    def get_bootsamples(self, num_samples, fix_initial=False):
        """
        Create bootstrap datasets using the y's predicted with the converged
        parameters. The final prediction error is sampled with replacement
        and used to generate artificial datasets, wich are all subjected to
        optimization by the bootstrap_params method below

        Parameters
        ----------
        num_samples : int
            number of bootstrap samples.
        fix_initial : bool, optional
            If True, the initial y values are subjected to error.
            The default is False.

        Returns
        -------
        y_boot : list of lists
            each internal list contains num_samples datasets of dimension
            n_x x n_states for each experimental dataset provided

        """
        # Remember that multiple datasets are allowed

        y_boot = []
        for ind in range(self.inst.num_datasets):  # datasets
            residual = self.residuals[ind].T
            y_nominal = self.y_nominal[ind].T

            y_states = []
            for res, y in zip(residual, y_nominal):  # states
                resid = res[fix_initial:]

                boots = np.random.choice(resid,
                                         size=(num_samples, len(resid)),
                                         replace=True)

                y_generated = y[fix_initial:] - boots

                if fix_initial:
                    y_generated = np.insert(y_generated, 0, y[0], axis=1)

                y_states.append(y_generated)

            y_experiment = [np.column_stack(a) for a in list(zip(*y_states))]

            y_boot.append(y_experiment)

        return y_boot

    def bootstrap_params(self, num_samples=100):
        y_samples = self.get_bootsamples(num_samples)

        # Run parameter estimation
        boot_params = np.zeros((num_samples, self.inst.num_params))

        tic = time.time()
        for ind in range(num_samples):
            # Update bootstraped data for all the experimental runs
            self.inst.y_data = [y[ind] for y in y_samples]

            # Optimize
            try:
                params, hess_inv, info = self.inst.optimize_fn(
                    method=self.inst.opt_method,
                    verbose=False, store_iter=False,
                    optim_options=self.inst.optim_options)
            except:
                print('Optimization failed.')
                params = [np.nan] * self.inst.num_params

            # Store
            boot_params[ind] = params

        toc = time.time()

        self.boot_params = boot_params
        self.num_samples = num_samples

        elapsed = toc - tic

        print()
        print('Bootstrap time: {:.2f} s'.format(elapsed))
        print()

        return boot_params

    def plot_hist(self, fig_size=None):

        num_plots = self.inst.num_params
        num_cols = bool(num_plots // 2) + 1
        num_rows = num_plots // 2 + num_plots % 2

        if num_rows == 1:
            mult = 1.2
        else:
            mult = 1.0

        if fig_size is None:
            width = 5  # in
            height = width/num_cols/mult * num_rows
            fig_size = (width, height)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
        for ind, ax in enumerate(axes.flatten()):
            ax.hist(self.boot_params.T[ind], bins=50)
            ax.set_xlabel(self.inst.param_names_plot[ind])
            ax.axvline(self.params[ind], linestyle='--', color='k')
            ylims = ax.get_ylim()

            ax.text(self.params[ind], ylims[1]*1.1,
                    '{:.3f}'.format(self.params[ind]), ha='center',
                    fontsize=8)

        fig.text(-0.02, 0.5,
                 r'Frequency (%i samples)' % self.num_samples,
                 rotation=90, va='center')
        fig.tight_layout()

        return fig, axes

    def confid_profiles(sens, hess_inv_params, num_times, variance_resid,
                        alpha, dof):

        sigma_resid = np.sqrt(variance_resid)
        sens_ord = []
        for ind in range(num_times):
            sens_time = sens[ind::num_times]
            sens_ord.append(sens_time)

        num_states = len(sens_ord[0])
        sigma_y = np.zeros((num_times, num_states))

        for ind in range(num_times):
            cov_y = (sens_ord[ind].dot(hess_inv_params)).dot(sens_ord[ind].T)
            sigma_y[ind] = np.sqrt(np.diag(cov_y)) * sigma_resid

        t_stat = stats.t.ppf((1 + alpha)/2, dof)

        sigma_y *= t_stat

        return sigma_y, sens_ord

    def plot_boots(self, num_samples, ind, fig_size=None):
        bootsamples = self.get_bootsamples(num_samples)
        num_div = bootsamples[ind].shape[1] // self.inst.num_xs[ind]
        y_data = np.split(bootsamples[ind], num_div, axis=1)

        nplot = len(y_data)
        ncol = 2
        nrow = nplot // ncol + nplot % ncol

        fig, axes = plt.subplots(nrow, ncol, figsize=fig_size)

        axes = np.atleast_1d(axes)

        for ind, data in enumerate(y_data):
            axes.flatten()[ind].plot(self.inst.x_data[0], data.T, '-o',
                                     mfc='None')

        if nplot < len(axes.flatten()):
            fig.delaxes(axes.flatten()[-1])
        return fig, axes
