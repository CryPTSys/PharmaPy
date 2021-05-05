#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:35:48 2019

@author: casas100
"""

# from reactor_module import ReactorClass
import numpy as np
from scipy.linalg import svd
from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
from PharmaPy.jac_module import numerical_jac_data
from PharmaPy import Gaussians as gs

from PharmaPy.LevMarq import levenberg_marquardt
from PharmaPy.Commons import plot_sens

from itertools import cycle
from ipopt import minimize_ipopt

linestyles = cycle(['-', '--', '-.', ':'])

eps = np.finfo(float).eps


def pseudo_inv(states, dstates_dparam=None):
    # Pseudo-inverse
    U, di, VT = np.linalg.svd(states)
    D_plus = np.zeros_like(states).T
    np.fill_diagonal(D_plus, 1 / di)
    pseudo_inv = np.dot(VT.T, D_plus).dot(U.T)

    if dstates_dparam is None:
        return pseudo_inv, di
    else:
        # Derivative
        first_term = -pseudo_inv @ dstates_dparam @ pseudo_inv
        second_term = pseudo_inv @ pseudo_inv.T @ dstates_dparam.T @ \
            (1 - states @ pseudo_inv)
        third_term = (1 - pseudo_inv @ states) @ (pseudo_inv.T @ pseudo_inv)

        dplus_dtheta = first_term + second_term + third_term

        return pseudo_inv, di, dplus_dtheta


def mcr_spectra(conc, spectra):
    conc_plus, _ = pseudo_inv(conc)

    absortivity_pred = np.dot(conc_plus, spectra)
    absorbance_pred = np.dot(conc, absortivity_pred)

    return conc_plus, absortivity_pred, absorbance_pred


class Experiment:
    def __init__(self):
        pass


class ParameterEstimation:

    """ Create a ParameterEstimation object

    Parameters
    ----------

    func : callable
        model function. Its signaure must be func(params, x_data, *args)
    params_seed : array-like
        parameter seed values
    x_data : numpy array or list of arrays
        1 x num_data array with experimental values for the independent
        variable.
    y_data : numpy array or list of arrays
        n_states x num_data experimental values for the dependent variable(s)
    """

    def __init__(self, func, param_seed, x_data, y_data=None,
                 args_fun=None,
                 optimize_flags=None,
                 jac_fun=None, dx_finitediff=None,
                 measured_ind=None, covar_data=None,
                 name_params=None, name_states=None):

        self.function = func

        self.jac_fun = jac_fun
        self.dx_fd = dx_finitediff

        self.fit_spectra = False

        param_seed = np.asarray(param_seed)
        if param_seed.ndim == 0:
            param_seed = param_seed[np.newaxis]

        # --------------- Data
        self.measured_ind = measured_ind

        if isinstance(x_data, dict) and isinstance(y_data, dict):

            keys_x = x_data.keys()
            keys_y = y_data.keys()

            if keys_x == keys_y:
                keys_dict = keys_x
            else:
                raise NameError("Keys of the x_data and y_data dictionaries "
                                "don't match")

            x_fit = []
            y_fit = []
            x_match = []

            x_exp = []
            y_exp = []
            for key in keys_dict:
                x_vals, y_vals, x_common, y_experim = self.interpret_data(
                    x_data[key], y_data[key])

                x_fit.append(x_vals)
                y_fit.append(y_vals)
                x_match.append(x_common)

                x_exp.append(x_data[key])
                y_exp.append(y_experim)

        else:
            x_fit, y_fit, x_common, y_experim = self.interpret_data(
                x_data, y_data)

            x_fit = [x_fit]
            y_fit = [y_fit]
            x_match = [x_common]

            x_exp = [x_data]
            y_exp = [y_experim]

        self.x_data = x_exp
        self.y_data = y_exp
        self.x_match = x_match

        self.x_fit = x_fit
        self.y_fit = y_fit

        # ---------- Covariance
        if (covar_data is not None) and (type(covar_data) is not list):
            covar_data = [covar_data]

        self.num_datasets = len(self.y_fit)

        # ---------- Arguments
        if args_fun is None:
            args_fun = [()] * self.num_datasets
        elif self.num_datasets == 1:
            args_fun = [args_fun]

        self.args_fun = args_fun

        self.num_xs = np.array([len(xs) for xs in self.x_fit])
        self.num_data = [array.size for array in self.y_fit]
        self.num_data_total = sum(self.num_data)

        if covar_data is None:
            self.stdev_data = [np.ones(num_data) for num_data in self.num_data]
        else:
            self.stdev_data = [np.sqrt(covar.T.ravel())
                               for covar in covar_data]

        # --------------- Parameters
        self.num_params_total = len(param_seed)
        if optimize_flags is None:
            self.map_fixed = []
            self.map_variable = np.array([True]*self.num_params_total)
        else:
            self.map_variable = np.array(optimize_flags)
            self.map_fixed = ~self.map_variable

        self.param_seed = param_seed

        self.num_params = self.map_variable.sum()

        # --------------- Names
        # Parameters
        if name_params is None:
            self.name_params = ['theta_{}'.format(ind + 1)
                                for ind in range(self.num_params)]

            self.name_params_total = ['theta_{}'.format(ind + 1)
                                      for ind in range(self.num_params_total)]

            self.name_params_plot = [r'$\theta_{}$'.format(ind + 1)
                                     for ind in range(self.num_params)]
        else:
            self.name_params_total = name_params
            self.name_params = [name_params[ind]
                                for ind in range(len(name_params))
                                if self.map_variable[ind]]
            self.name_params_plot = [r'$' + name + '$'
                                     for name in self.name_params]

        # ---------- States
        if name_states is None:
            self.name_states = [r'$y_{}$'.format(ind + 1)
                                for ind in range(self.num_states)]
        else:
            self.name_states = name_states

        # --------------- Outputs
        self.params_iter = []
        self.objfun_iter = []
        self.cond_number = []

        self.resid_runs = None
        self.y_runs = None
        self.sens = None
        self.sens_runs = None
        self.y_model = []

    def interpret_data(self, x_data, y_data):
        x_match = None

        if isinstance(x_data, (list, tuple)):  # several datasets
            # args_fun = (args_fun, )

            x_fit = np.concatenate(x_data)
            x_fit.sort()
            x_fit = np.unique(x_fit)

            x_match = [np.isin(x_fit, array) for array in x_data]

            x_fit = x_fit
            y_fit = np.concatenate(y_data)

            y_data = [item[..., np.newaxis] for item in y_data
                      if item.ndim == 1]

        else:  # unique dataset
            x_fit = x_data

            if y_data.ndim == 1:
                y_data = y_data[..., np.newaxis]

            y_data = y_data.T

            y_fit = y_data.flatten()

        self.num_states = len(y_data)

        if self.measured_ind is None:
            self.measured_ind = np.arange(0, self.num_states)

        self.num_measured = len(self.measured_ind)

        return x_fit, y_fit, x_match, y_data

    def scale_sens(self, param_lims=None):
        """ Scale sensitivity matrix to make it non-dimensional.
        After Brun et al. Water Research, 36, 4113-4127 (2002),
        Jorke et al. Chem. Ing. Tech. 2015, 87, No. 6, 713-725,
        McLean et al. Can. J. Chem. Eng. 2012, 90, 351-366

        """

        ord_sens = self.reorder_sens(separate_sens=True)
        selected_sens = [ord_sens[ind] for ind in self.measured_ind]

        if param_lims is None:
            for ind, sens in enumerate(selected_sens):
                conc_time = self.conc_profile[:, ind][..., np.newaxis]
                sens *= self.params / conc_time
        else:
            for ind, sens in enumerate(selected_sens):
                delta_param = [par[1] - par[0] for par in param_lims]
                delta_param = np.array(delta_param)
                sens *= delta_param / self.delta_conc[ind]

        return np.vstack(selected_sens)

    def select_sens(self, sens_ordered, num_xs, times=None):
        parts = np.vsplit(sens_ordered, len(sens_ordered)//num_xs)

        if times is None:
            selected = [parts[ind] for ind in self.measured_ind]
        else:
            selected = [parts[ind][times[count]]
                        for count, ind in enumerate(self.measured_ind)]

        selected_array = np.vstack(selected)

        return selected_array

    def reconstruct_params(self, params):
        params_reconstr = np.zeros(self.num_params_total)
        params_reconstr[self.map_fixed] = self.param_seed[self.map_fixed]
        params_reconstr[self.map_variable] = params

        return params_reconstr

    def func_aux(self, params, x_vals, args):
        states = self.function(params, x_vals, *args)

        return states.T.ravel()

    def get_objective(self, params, residual_vec=False):
        # Reconstruct parameter set with fixed and non-fixed indexes
        params = self.reconstruct_params(params)

        # Store parameter values
        if type(self.params_iter) is list:
            self.params_iter.append(params)

        # --------------- Solve
        y_runs = []
        resid_runs = []
        sens_runs = []

        for ind in range(self.num_datasets):
            # Solve
            result = self.function(params, self.x_fit[ind],
                                   *self.args_fun[ind])

            if type(result) is tuple:  # func also returns the jacobian
                y_prof, sens = result

            else:  # call a separate function for jacobian
                y_prof = result

                if self.jac_fun is None:
                    sens = numerical_jac_data(
                        self.func_aux, params,
                        (self.x_fit[ind], self.args_fun[ind]),
                        dx=self.dx_fd)
                else:
                    sens = self.jac_fun(params, self.x_fit[ind],
                                        *self.args_fun[ind])

            if y_prof.ndim == 1:
                y_run = y_prof
                sens_run = sens
            else:
                y_run = y_prof[:, self.measured_ind]

                if self.x_match[ind] is None:
                    y_run = y_run.T.ravel()
                    sens_run = self.select_sens(sens, self.num_xs[ind])
                else:
                    y_run = [y_run[idx, col]
                             for col, idx in enumerate(self.x_match[ind])]
                    y_run = np.concatenate(y_run)

                    x_sens = self.x_match[ind]
                    sens_run = self.select_sens(sens, self.num_xs[ind], x_sens)

            resid_run = (y_run - self.y_fit[ind])/self.stdev_data[ind]

            # Store
            y_runs.append(y_run)
            resid_runs.append(resid_run)
            sens_runs.append(sens_run)

        self.sens_runs = sens_runs
        self.y_runs = y_runs
        self.resid_runs = resid_runs

        if type(self.objfun_iter) is list:
            objfun_val = np.linalg.norm(np.concatenate(self.resid_runs))**2
            self.objfun_iter.append(objfun_val)

        residuals = self.optimize_flag * np.concatenate(resid_runs)
        self.residuals = residuals

        # Return objective
        if residual_vec:
            return residuals
        else:
            residual = 1/2 * residuals.dot(residuals)
            return residual

    def get_gradient(self, params, jac_matrix=False):
        if self.sens_runs is None:  # TODO: this is a hack to allow IPOPT
            self.objective_fun(params)

        concat_sens = np.vstack(self.sens_runs)
        if not self.fit_spectra:
            concat_sens = concat_sens[:, self.map_variable]

        self.sens = concat_sens

        # if type(self.cond_number) is list:
        #     self.cond_number.append(self.get_cond_number(concat_sens))

        std_dev = np.concatenate(self.stdev_data)
        jacobian = (concat_sens.T / std_dev)  # 2D

        if jac_matrix:
            return jacobian
        else:
            gradient = jacobian.dot(self.residuals)  # 1D
            return gradient

    def get_cond_number(self, sens_matrix):
        _, sing_vals, _ = np.linalg.svd(sens_matrix)

        cond_number = max(sing_vals) / min(sing_vals)

        return cond_number

    def optimize_fn(self, optim_options=None, simulate=False, verbose=True,
                    store_iter=True, method='LM', bounds=None):

        self.optimize_flag = not simulate
        params_var = self.param_seed[self.map_variable]

        if method == 'LM':
            self.opt_method = 'LM'
            if optim_options is None:
                optim_options = {'full_output': True, 'verbose': verbose}
            else:
                optim_options['full_output'] = True
                optim_options['verbose'] = verbose

            opt_par, inv_hessian, info = levenberg_marquardt(
                params_var,
                self.get_objective,
                self.get_gradient,
                args=(True,),
                **optim_options)

        elif method == 'IPOPT':
            self.opt_method = 'IPOPT'
            if optim_options is None:
                optim_options = {'print_level': int(verbose) * 5}
            else:
                optim_options['print_level'] = int(verbose) * 5

            result = minimize_ipopt(self.get_objective, params_var,
                                    jac=self.get_gradient,
                                    bounds=bounds, options=optim_options)

            opt_par = result['x']

            final_sens = np.vstack(self.sens_runs)[:, self.map_variable].T
            final_fun = np.concatenate(self.resid_runs)
            info = {'jac': final_sens, 'fun': final_fun}

        self.optim_options = optim_options

        # Store
        self.params_convg = opt_par
        # self.covar_params = inv_hessian
        self.info_opt = info

        self.cond_number = np.array(self.cond_number)
        self.params_iter = np.array(self.params_iter)
        _, idx = np.unique(self.params_iter, axis=0, return_index=True)

        if store_iter:
            self.params_iter = self.params_iter[np.sort(idx)]
            self.objfun_iter = np.array(self.objfun_iter)[np.sort(idx)]

            col_names = ['obj_fun'] + self.name_params_total
            self.paramest_df = pd.DataFrame(
                np.column_stack((self.objfun_iter, self.params_iter)),
                columns=col_names)

        # Model prediction with final parameters
        for ind in range(self.num_datasets):
            y_model_flat = self.resid_runs[ind]*self.stdev_data[ind] + \
                self.y_fit[ind]

            if self.x_match[ind] is None:
                y_reshape = y_model_flat.reshape(-1, self.num_xs[ind]).T
            else:
                x_len = [sum(array) for array in self.x_match[ind]]
                x_sum = np.cumsum(x_len)[:-1]
                y_reshape = np.split(y_model_flat, x_sum)

            self.y_model.append(y_reshape)

        covar_params = self.get_covariance()

        return opt_par, covar_params, info

    def get_covariance(self):
        jac = self.info_opt['jac']
        resid = self.info_opt['fun']

        hessian_approx = np.dot(jac, jac.T)

        dof = self.num_data_total - self.num_params
        mse = 1 / dof * np.dot(resid, resid)

        covar = mse * np.linalg.inv(hessian_approx)

        # Correlation matrix
        sigma = np.sqrt(covar.diagonal())
        d_matrix = np.diag(1/sigma)
        correlation = d_matrix.dot(covar).dot(d_matrix)

        self.covar_params = covar
        self.correl_params = correlation

        return covar

    def inspect_data(self, fig_size=None):
        states_seed = []

        if self.fit_spectra:
            kwarg_sens = {'reorder': False}
        else:
            kwarg_sens = {}

        for ind in range(self.num_datasets):
            states_pred = self.function(self.param_seed, self.x_fit[ind],
                                           *self.args_fun[ind],
                                           **kwarg_sens)

            if isinstance(states_pred, tuple):
                states_pred = states_pred[0]

            states_seed.append(states_pred)

        if len(states_seed) == 1:
            y_seed = states_seed[0]

            x_data = self.x_data
            y_data = self.y_data

            fig, axes = plt.subplots(len(y_data), figsize=fig_size)

            axes = np.atleast_1d(axes)

            for ind, experimental in enumerate(y_data):
                axes[ind].plot(x_data[ind], y_seed[:, self.measured_ind])

                markers = cycle(['o', 's', '^', '*', 'P', 'X'])

                for idx, row in enumerate(experimental):
                    color = axes[ind].lines[idx].get_color()
                    axes[ind].plot(x_data[ind], row, lw=0,
                                   marker=next(markers), ms=5,
                                   mfc='None', color=color)

                axes[ind].spines['right'].set_visible(False)
                axes[ind].spines['top'].set_visible(False)

                axes[ind].set_ylabel(self.name_states[ind])

            axes[0].lines[0].set_label('prediction with seed parameters')
            axes[0].lines[self.num_measured].set_label('data')

            axes[0].legend(loc='best')
            axes[-1].set_xlabel('$x$')

        else:
            pass  # TODO what to do with multiple datasets, maybe a parity plot?

    def plot_data_model(self, fig_size=None, fig_grid=None, fig_kwargs=None,
                        plot_initial=False, black_white=False):

        num_plots = self.num_datasets

        if fig_grid is None:
            num_cols = bool(num_plots // 2) + 1
            num_rows = num_plots // 2 + num_plots % 2
        else:
            num_cols = fig_grid[1]
            num_rows = fig_grid[0]

        fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)

        if num_plots == 1:
            axes = np.asarray(axes)[np.newaxis]

        if fig_kwargs is None:
            fig_kwargs = {'mfc': 'None', 'ls': '', 'ms': 4}

        ax_flatten = axes.flatten()

        x_data = self.x_data
        y_data = self.y_data

        # if any([item is not None for item in self.x_match]):
        #     raise NotImplementedError('One or more datasets of type 2.')

        for ind in range(self.num_datasets):  # experiment loop
            markers = cycle(['o', 's', '^', '*', 'P', 'X'])

            # Model prediction
            if black_white:
                ax_flatten[ind].plot(x_data[ind], self.y_model[ind], 'k')

                for col in y_data[ind]:
                    ax_flatten[ind].plot(x_data[ind], col, color='k',
                                         marker=next(markers), **fig_kwargs)
            else:
                try:  # Type 1
                    ax_flatten[ind].plot(x_data[ind], self.y_model[ind])
                except:  # Type 2
                    for x, y in zip(x_data[ind], self.y_model[ind]):
                        ax_flatten[ind].plot(x, y)

                lines = ax_flatten[ind].lines
                colors = [line.get_color() for line in lines]

                try:  # Type 1
                    for color, col in zip(colors, y_data[ind]):
                        ax_flatten[ind].plot(x_data[ind], col, color=color,
                                             marker=next(markers),
                                             **fig_kwargs)
                except:  # Type 2
                    for color, x, y in zip(colors, x_data[ind], y_data[ind]):
                        ax_flatten[ind].plot(x, y, color=color,
                                             marker=next(markers),
                                             **fig_kwargs)


            # Edit
            ax_flatten[ind].spines['right'].set_visible(False)
            ax_flatten[ind].spines['top'].set_visible(False)

            ax_flatten[ind].set_xlabel('$x$')
            ax_flatten[ind].set_ylabel(r'$\mathbf{y}$')

            ax_flatten[ind].xaxis.set_minor_locator(AutoMinorLocator(2))
            ax_flatten[ind].yaxis.set_minor_locator(AutoMinorLocator(2))

        # ax_flatten[0].legend(names_meas, loc='best')

        if plot_initial:
            if self.x_match[ind] is None:  # type 1
                x_exp = [x_data[ind]] * len(self.y_model)
            else:  # type 2
                x_exp = x_data[ind]

            residuals_convg = self.residuals.copy()
            resid_runs_convg = self.resid_runs.copy()

            seed_params = self.param_seed[self.map_variable]
            resid_seed = self.get_objective(seed_params, residual_vec=True)

            idx_split = np.cumsum(self.num_measured * self.num_xs)[:-1]
            resid_seed = np.split(resid_seed, idx_split)

            for ind in range(self.num_datasets):
                x_exp = x_data[ind]

                ymodel_seed = resid_seed[ind] + self.y_fit[ind]
                ymodel_seed = ymodel_seed.reshape(-1, self.num_xs[ind])

                markers = cycle(['o', 's', '^', '*', 'P', 'X'])
                fig_kwargs['ls'] = '-'
                if black_white:
                    for rowind, col in enumerate(ymodel_seed):
                        ax_flatten[ind].plot(x_exp, col, '--',
                                             color='k',
                                             marker=next(markers), ms=3,
                                             alpha=0.3, **fig_kwargs)
                        # markevery=3)
                else:
                    for rowind, col in enumerate(ymodel_seed):
                        ax_flatten[ind].plot(x_exp, col, '--',
                                             marker=next(markers),
                                             color=colors[rowind],
                                             alpha=0.3, **fig_kwargs)
                        # markevery=3)

                self.resid_runs = resid_runs_convg
                self.residuals = residuals_convg

        if len(ax_flatten) > self.num_datasets:
            fig.delaxes(ax_flatten[-1])

        if len(axes) == 1:
            axes = axes[0]

        fig.tight_layout()

        return fig, axes

    def plot_data_model_sep(self, fig_size=None, fig_kwargs=None,
                            plot_initial=False):

        if len(self.x_fit) > 1:
            raise NotImplementedError('More than one dataset detected. '
                                      'Not supported for multiple datasets')

        if self.x_match[0] is None:
            ydata = self.y_data[0]
            xdata = [self.x_data[0]] * len(ydata)
            ymodel = self.y_model[0].T  # TODO: change this
        else:
            xdata = self.x_data
            ydata = self.y_data
            ymodel = self.y_model[0]

        num_x = self.num_xs[0]
        num_plots = self.num_states

        num_col = 2
        num_row = num_plots // num_col + num_plots % num_col

        num_row = 2
        num_col = num_plots // num_row + num_plots % num_row

        fig, axes = plt.subplots(num_row, num_col, figsize=fig_size)

        for ind in range(num_plots):  # for every state
            axes.flatten()[ind].plot(xdata[ind], ymodel[ind])

            line = axes.flatten()[ind].lines[0]
            color = line.get_color()
            axes.flatten()[ind].plot(xdata[ind], ydata[ind].T, 'o', color=color,
                                     mfc='None')

            axes.flatten()[ind].set_ylabel(self.name_states[ind])

        if plot_initial:
            residuals_convg = self.residuals.copy()
            resid_runs_convg = self.resid_runs.copy()

            seed_params = self.param_seed[self.map_variable]
            resid_seed = self.get_objective(seed_params, residual_vec=True)

            ymodel_seed = resid_seed*self.stdev_data[0] + self.y_fit[0]

            if self.x_match[0] is None:
                ymodel_seed = ymodel_seed.reshape(-1, num_x)
            else:
                x_len = [sum(array) for array in self.x_match]
                x_sum = np.cumsum(x_len)[:-1]
                ymodel_seed = np.split(ymodel_seed, x_sum)

            for ind in range(num_plots):
                axes.flatten()[ind].plot(xdata[ind], ymodel_seed[ind], '--',
                                         color=color,
                                         alpha=0.4)

            self.resid_runs = resid_runs_convg
            self.residuals = residuals_convg

        # fig.text(0.5, 0, '$x$', ha='center')
        fig.tight_layout()
        return fig, axes

    def plot_sens_param(self):
        figs = []
        axes = []
        for times, sens in zip(self.x_data, self.sens_runs):
            fig, ax = plot_sens(times, sens)

            figs.append(fig)
            axes.append(ax)

        return figs, axes

    def plot_parity(self, fig_size=(4.5, 4.0), fig_kwargs=None):
        if fig_kwargs is None:
            fig_kwargs = {'alpha': 0.70}

        fig, axis = plt.subplots(figsize=fig_size)

        if all([item is None for item in self.x_match]):
            y_model = self.y_model
        else:
            y_model = [np.concatenate(lst) for lst in self.y_model]

        for ind, y_model in enumerate(y_model):
            axis.scatter(y_model.T.flatten(), self.y_fit[ind],
                         label='experiment {}'.format(ind + 1),
                         **fig_kwargs)

        axis.set_xlabel('Model')
        axis.set_ylabel('Data')
        axis.legend(loc='best')

        all_data = np.concatenate((np.concatenate(self.y_runs),
                                   np.concatenate(self.y_fit)))
        plot_min = min(all_data)
        plot_max = max(all_data)

        offset = 0.05*plot_max
        x_central = [plot_min - offset, plot_max + offset]

        axis.plot(x_central, x_central, 'k')
        axis.set_xlim(x_central)
        axis.set_ylim(x_central)

        return fig, axis

    def plot_correlation(self):

        # Mask
        mask = np.tri(self.num_params, k=-1).T
        corr_masked = np.ma.masked_array(self.correl_params, mask=mask)

        # Plot
        fig_heat, axis_heat = plt.subplots()

        heatmap = axis_heat.imshow(corr_masked, cmap='RdBu', aspect='equal',
                                   vmin=-1, vmax=1)
        divider = make_axes_locatable(axis_heat)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig_heat.colorbar(heatmap, ax=axis_heat, cax=cax)
        cbar.outline.set_visible(False)

        axis_heat.set_xticks(range(self.num_params))
        axis_heat.set_yticks(range(self.num_params))

        axis_heat.set_xticklabels(self.name_params_plot, rotation=90)
        axis_heat.set_yticklabels(self.name_params_plot)

        return fig_heat, axis_heat


class Deconvolution:
    def __init__(self, mu, sigma, ampl, x_data, y_data):

        self.mu = mu
        self.sigma = sigma
        self.ampl = ampl

        self.x_data = x_data
        self.y_data = y_data

    def concat_params(self):
        if isinstance(self.mu, float) or isinstance(self.mu, int):
            params_concat = np.array([self.mu, self.sigma, self.ampl])
        else:
            params_concat = np.concatenate((self.mu, self.sigma, self.ampl))

        return params_concat

    def fun_wrapper(self, params, x_data):

        grouped_params = np.split(params, 3)
        gaussian = gs.multiple_gaussian(self.x_data, *grouped_params)

        return gaussian

    def dparam_wrapper(self, params, x_data):
        grouped_params = np.split(params, 3)
        der_params = gs.gauss_dparam_mult(x_data, *grouped_params)

        return der_params

    def dx_wrapper(self, params, x_data):
        grouped_params = np.split(params, 3)
        der_x = gs.gauss_dx_mult(x_data, *grouped_params)

        return der_x

    def inspect_data(self):
        gaussian_pred = gs.multiple_gaussian(self.x_data, self.mu, self.sigma,
                                             self.ampl)

        fig, axis = plt.subplots()

        axis.plot(self.x_data, gaussian_pred)
        axis.plot(self.x_data, self.y_data, 'o', mfc='None')

        axis.legend(('prediction with seed params', 'experimental data'))
        axis.set_xlabel('$x$')
        axis.set_ylabel('signal')

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        return fig, axis, gaussian_pred

    # def estimate_params(self, optim_opt=None):
    #     seed = self.concat_params()

    #     paramest = ParameterEstimation(self.fun_wrapper, seed, self.x_data,
    #                                    self.y_data,
    #                                    df_dtheta=self.dparam_wrapper,
    #                                    df_dy=self.dx_wrapper)

    #     result = paramest.optimize_fn(optim_options=optim_opt)

    #     optim_params = np.split(result[0], 3)

    #     self.param_obj = paramest
    #     self.optim_params = optim_params

    #     return optim_params, result[1:], paramest

    def plot_results(self, fig_size=None, plot_initial=False,
                     plot_individual=False):

        fig, axis = self.param_obj.plot_data_model(fig_size=fig_size,
                                                   plot_initial=plot_initial)

        axis.legend(('best fit', 'experimental data'))
        axis.set_ylabel('signal')

        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        if plot_individual:
            gaussian_vals = gs.multiple_gaussian(
                self.x_data, *self.optim_params, separated=True)

            for row in gaussian_vals.T:
                axis.plot(self.x_data, row, '--')
                color = axis.lines[-1].get_color()

                axis.fill_between(self.x_data, row.min(), row, fc=color,
                                  alpha=0.2)

        return fig, axis

    def plot_deriv(self, which='both', plot_mu=False):
        fig, axis = plt.subplots()

        # fun = gs.multiple_gaussian(self.x_data, *self.optim_params)

        if which == 'both':
            first = gs.gauss_dx_mult(self.x_data, *self.optim_params)
            second = gs.gauss_dxdx_mult(self.x_data, *self.optim_params)

            axis.plot(self.x_data, first)
            axis.set_ylabel('$\partial f / \partial x$')

            axis_sec = axis.twinx()
            axis_sec.plot(self.x_data, second, '--')
            axis_sec.set_ylabel('$\partial^2 f / \partial x^2$')

            fig.legend(('first', 'second'), bbox_to_anchor=(1, 1),
                       bbox_transform=axis.transAxes)

            axis.spines['top'].set_visible(False)
            axis_sec.spines['top'].set_visible(False)

        elif which == 'first':
            first = gs.gauss_dx_mult(self.x_data, *self.optim_params)
            axis.plot(self.x_data, first)
            axis.set_ylabel('$\partial f / \partial x$')

            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)

        elif which == 'second':
            second = gs.gauss_dxdx_mult(self.x_data, *self.optim_params)
            axis.plot(self.x_data, second)
            axis.set_ylabel('$\partial^2 f / \partial x^2$')

            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)

        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        axis.set_xlabel('$x$')

        if plot_mu:
            mu_opt = self.optim_params[0]

            for mu in mu_opt:
                axis.axvline(mu, ls='--', alpha=0.4)

        return fig, axis


class MultipleCurveResolution(ParameterEstimation):
    def __init__(self, func, param_seed, time_data, spectra=None,
                 args_fun=None,
                 optimize_flags=None,
                 jac_fun=None, dx_finitediff=None,
                 measured_ind=None, covar_data=None,
                 name_params=None, name_states=None):

        super().__init__(func, param_seed, time_data, spectra,
                         args_fun, optimize_flags, jac_fun,
                         dx_finitediff, measured_ind, covar_data,
                         name_params, name_states)

        self.fit_spectra = True
        self.spectra = [data.T for data in self.y_data]

    def get_objective(self, params, residual_vec=False):
        if type(self.params_iter) is list:
            self.params_iter.append(params)

        def func_aux(params, x_vals, spectra, *args):
            states = self.function(params, x_vals, *args)

            _, epsilon, absorbance = mcr_spectra(
                states[:, self.measured_ind], spectra)

            return absorbance.T.ravel()

        y_runs = []
        resid_runs = []
        sens_runs = []

        for ind in range(self.num_datasets):
            result = self.function(params, self.x_fit[ind],
                                   reorder=False,
                                   *self.args_fun[ind])

            if isinstance(result, tuple):
                conc_prof, sens_states = result
            else:
                conc_prof = result
                sens_states = None

            conc_target = conc_prof[:, self.measured_ind]

            # MCR
            conc_plus = np.linalg.pinv(conc_target)
            absortivity_pred = np.dot(conc_plus, self.spectra[ind])
            spectra_pred = np.dot(conc_target, absortivity_pred)

            self.epsilon_mcr = absortivity_pred

            if sens_states is None:
                args_merged = [self.x_data[ind],
                               self.args_fun[ind], self.spectra[ind]]
                sens = numerical_jac_data(func_aux, params, args_merged,
                                          dx=self.dx_fd)[:, self.map_variable]
            else:
                eye = np.eye(conc_target.shape[0])
                proj_orthogonal = eye - np.dot(conc_target, conc_plus)

                sens_pick = sens_states[self.map_variable][
                    :, :, self.measured_ind]

                first_term = proj_orthogonal @ sens_pick @ conc_plus
                second_term = first_term.transpose((0, 2, 1))
                sens_an = (first_term + second_term) @ spectra_pred

                n_par, n_times, n_conc = sens_an.shape
                sens = sens_an.T.reshape(n_conc * n_times, n_par)

            resid = (spectra_pred - self.spectra[ind]).T.ravel()
            resid_run = resid / self.stdev_data[ind]

            y_runs.append(spectra_pred.T.ravel())
            resid_runs.append(resid_run)
            sens_runs.append(sens)

        self.sens_runs = sens_runs
        self.y_runs = y_runs
        self.resid_runs = resid_runs

        if type(self.objfun_iter) is list:
            objfun_val = np.linalg.norm(np.concatenate(self.resid_runs))**2
            self.objfun_iter.append(objfun_val)

        residuals = self.optimize_flag * np.concatenate(resid_runs)
        self.residuals = residuals

        # Return objective
        if residual_vec:
            return residuals
        else:
            residual = 1/2 * residuals.dot(residuals)
            return residual


if __name__ == '__main__':
    import englezos_example as englezos

    # Data
    data = np.genfromtxt('../data/englezos_example.csv', delimiter=',',
                         skip_header=1)
    t_exp, c3_exp = data.T

    init_conc = [60, 60, 0]
    param_seed = [1e-5, 1e-5]
#    param_seed = [0.4577e-5, 0.2797e-3]

    reaction_matrix = np.array([-2, -1, 2])
    species = ('$NO$', '$O_2$', '$NO_2$')

    param_object = ParameterEstimation(
        reaction_matrix, param_seed,
        t_exp, c3_exp,
        y_init=init_conc,
        measured_ind=(-1,),
        kinetic_model=englezos.bodenstein_linder,
        df_dstates=englezos.jac_conc,
        df_dtheta=englezos.jac_par,
        names_species=species)

    simulate = True

    if simulate:
        param_object.solve_model(init_conc, x_eval=t_exp, eval_sens=True)
        param_object.plot_states()
        fig_sens, axes_sens = param_object.plot_sens(fig_size=(5, 2))

        sens_total = param_object.reorder_sens()
        U, sing_vals, V = svd(sens_total)
        cond_number = max(sing_vals) / min(sing_vals)

        labels = list('ab')
        for ax, lab in zip(axes_sens, labels):
            ax.text(0.05, 0.9, lab, transform=ax.transAxes)

        fig_sens.savefig('../img/sens_englezos.pdf', bbox_inches='tight')

    else:
        optim_options = {'max_iter': 150, 'full_output': True, 'tau': 1e-2}

        params_optim, covar, info = param_object.optimize_fn(
            optim_options=optim_options)
        param_object.plot_data_model()
