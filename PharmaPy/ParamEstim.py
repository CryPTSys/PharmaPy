#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:35:48 2019

@author: casas100
"""

# from reactor_module import ReactorClass
import numpy as np
from scipy.linalg import svd, inv, ldl
from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
from PharmaPy.jac_module import numerical_jac_data, dx_jac_p
from PharmaPy import Gaussians as gs

from PharmaPy.LevMarq import levenberg_marquardt
from PharmaPy.Commons import plot_sens, reorder_sens

from cyipopt import minimize_ipopt

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


def get_masked_ydata(y_list, masks, assign_missing=None):
    nrow, ncol = masks.shape
    y_out = np.zeros((nrow, ncol))

    for col, mask in enumerate(masks.T):
        y_out[mask, col] = y_list[col]

        if assign_missing is not None:

            if isinstance(assign_missing, (np.ndarray, list)):
                y_out[~mask] = assign_missing[~mask]
            else:
                y_out[~mask] = assign_missing

    return y_out


def analyze_data(x_list, y_list):
    x_model = []
    x_mask = []

    y_masked = []
    for x_data, y_data in zip(x_list, y_list):
        if isinstance(x_data, (list, tuple)):
            x_all = np.sort(np.hstack(x_data))
            x_unique = np.unique(x_all)

            x_common = [np.isin(x_unique, data) for data in x_data]
            x_common = np.column_stack(x_common)

            x_model.append(x_unique)
            x_mask.append(x_common)

            y_mask = get_masked_ydata(y_data, x_common, np.nan)
            y_masked.append(y_mask)
        else:
            x_model.append(x_data)
            x_mask.append(None)

            y_masked.append(y_data)

    return x_model, x_mask, y_masked


class Experiment:
    def __init__(self):
        pass


class ParameterEstimation:

    def __init__(self, func, param_seed, x_data, y_data=None,
                 measured_ind=None,
                 args_fun=None, kwargs_fun=None,
                 optimize_flags=None,
                 jac_fun=None, dx_finitediff=None,
                 weight_matrix=None,
                 name_params=None, name_states=None):
        """ Create a ParameterEstimation object

        Parameters
        ----------
        func : callable
            model function with signaure func(params, x_data, *args, **kwargs).
            It must return either an array of size len(x_i) in the one-state
            case, and an array of size len(x_i) x num_states for models
            describing multiple states. See 'x_data' for details on x_i
        param_seed : array-like
            parameter seed values.
        x_data : numpy array, list of arrays or dict
            array with experimental values for the independent variable x. If
            several datasets Ne are passed, either a list of arrays
                x_data = [x_1, ..., x_i, ..., x_Ne]
            or a dictionary of arrays:
                x_data = {'name_exp_1': x_1,, ..., 'name_exp_i': x_i, ...,
                          'name_exp_N': x_Ne}
            can be specified.
        y_data : numpy array or list of arrays, optional
            experimental values for the dependent variable(s) y.
            Array y is of dimension len(x_i) x N_meas, where N_meas is less
            than or equal to the number of states returned by func (Ny).
            It supports same data structures as 'x_data'. If 'ydata' is a
            dictionary, its keys must match those of 'x_data'.
            The default is None.
        measured_ind : list of int, optional
            Indexes of the states returned by func that are measured and
            passed in each dataset contained in 'y_data'.
            If None, it is assumed that all the states are measured.
            The default is None.
        args_fun : tuple or list of tuples, optional
            positional arguments to be passed to func. For multiple datasets,
            pass a list of tuples. The default is None.
        kwargs_fun : dict, list of dicts, optional
            keyword arguments to be passed to func. For multiple datasets,
            pass a list of dicts. The default is None.
        optimize_flags : list of bools, optional
            list with dimension len(param_seed). If a given parameter is to
            be optimized, its corresponding flag is True. Otherwise, the flag
            must be False. If not provided, all the flags are set to
            True (all the parameters are used for optimization).
            The default is None.
        jac_fun : callable, optional
            jacobian function with the same signature as func. It must return
            an array with elements
                dy_j(t) / dparam_p (j = 1, ..., Ny, p = 1, ..., Np).

            The resulting array must be of size [sum_i len(x_i)] x num_params,
            which is formed by stacking jacobian matrices for each state
            vertically. If None, the jacobian is computed using finite
            differences. The default is None.
        dx_finitediff : float, optional
            perturbation in the parameter space used to estimate the
            parametric jacobian. The default is None.
        weight_matrix : numpy array, optional
            array with dimension N_meas x N_meas, indicating weighting
            factors for the measured states. A typical choice is a
            diagonal matrix of experimental state variances.
            The default is None.
        name_params : list of str, optional
            list with parameter names. The default is None.
        name_states : list of str, optional
            list with state names. The default is None.

        Returns
        -------
        ParameterEstimation object

        """

        self.function = func

        self.jac_fun = jac_fun

        self.fit_spectra = False

        param_seed = np.asarray(param_seed)
        if param_seed.ndim == 0:
            param_seed = param_seed[np.newaxis]

        if dx_finitediff is None:
            abs_tol = 1e-6 * np.ones_like(param_seed)
            rel_tol = 1e-6

            def dparam_fun(param):
                dp = dx_jac_p(param, abs_tol, rel_tol, eps)
                return dp

            dx_finitediff = dparam_fun

        self.dx_fd = dx_finitediff

        # --------------- Data
        self.experim_names = None

        if isinstance(x_data, dict) and isinstance(y_data, dict):
            self.experim_names = list(x_data.keys())

            x_data = list(x_data.values())
            y_data = list(y_data.values())

        elif isinstance(y_data, np.ndarray):
            x_data = [x_data]
            y_data = [y_data]

        y_data = [ar.reshape(-1, 1) if ar.ndim == 1 else ar for ar in y_data]

        x_model, x_masks, y_data = analyze_data(x_data, y_data)

        self.x_model = x_model
        self.x_masks = x_masks

        self.x_data = x_data
        self.y_data = y_data
        self.num_datasets = len(self.y_data)

        if self.experim_names is None:
            self.experim_names = ['exp_%i' % (ind + 1)
                                  for ind in range(self.num_datasets)]

        if measured_ind is None:
            measured_ind = list(range(y_data[0].shape[1]))

        self.measured_ind = measured_ind

        # ---------- Arguments
        if args_fun is None:
            args_fun = [()] * self.num_datasets
        elif self.num_datasets == 1:
            args_fun = [args_fun]
        else:
            args_fun = list(args_fun.values())

        if kwargs_fun is None:
            kwargs_fun = [{}] * self.num_datasets
        elif self.num_datasets == 1:
            # if not isinstance(kwargs_fun, list):
            kwargs_fun = [kwargs_fun]
        else:
            kwargs_fun = list(kwargs_fun.values())

        self.args_fun = args_fun
        self.kwargs_fun = kwargs_fun

        num_data = []
        for ind in range(self.num_datasets):
            if self.x_masks[ind] is None:
                num_data.append(self.y_data[ind].size)
            else:
                num_data.append(self.x_masks[ind].sum())

        self.num_data_total = sum(num_data)
        self.num_data = num_data

        if weight_matrix is None:
            weight_matrix = np.eye(len(self.measured_ind))

        l, d, perm = ldl(inv(weight_matrix))

        self.sigma_inv = np.dot(l[perm], d**0.5)

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
            if len(name_params) > sum(self.map_variable):
                self.name_params = [name_params[ind]
                                    for ind in range(len(name_params))
                                    if self.map_variable[ind]]
            else:
                self.name_params = name_params

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

    def select_sens(self, sens_ordered, num_states, times=None):

        parts = np.split(sens_ordered, num_states, axis=0)

        if times is None:
            selected = [parts[ind] for ind in self.measured_ind]
        else:
            selected = [parts[ind][times[count]]
                        for count, ind in enumerate(self.measured_ind)]

        return selected

    def reconstruct_params(self, params):
        params_reconstr = np.zeros(self.num_params_total)
        params_reconstr[self.map_fixed] = self.param_seed[self.map_fixed]
        params_reconstr[self.map_variable] = params

        return params_reconstr

    def func_aux(self, params, x_vals, args, kwargs):
        states = self.function(params, x_vals, *args, **kwargs)

        return states.T.ravel()

    def get_objective(self, params, residual_vec=False, set_self=True):
        # Store parameter values
        if type(self.params_iter) is list:
            self.params_iter.append(params)

        # Reconstruct parameter set with fixed and non-fixed indexes
        params = self.reconstruct_params(params)

        # --------------- Solve
        y_runs = []
        resid_runs = []
        sens_runs = []

        for ind in range(self.num_datasets):
            # Solve
            result = self.function(params, self.x_model[ind],
                                   *self.args_fun[ind], **self.kwargs_fun[ind])

            if isinstance(result, (tuple, list)):  # func also returns the jacobian
                y_prof, sens = result

            else:  # call a separate function for jacobian
                y_prof = result

                if self.jac_fun is None:
                    pass_to_fun = (self.x_model[ind], self.args_fun[ind],
                                   self.kwargs_fun[ind])

                    pick_p = np.where(self.map_variable)[0]
                    sens = numerical_jac_data(self.func_aux, params,
                                              pass_to_fun, dx=self.dx_fd,
                                              pick_x=pick_p)
                else:
                    sens = self.jac_fun(params, self.x_model[ind],
                                        *self.args_fun[ind],
                                        **self.kwargs_fun[ind])

            if y_prof.ndim == 1:
                y_run = y_prof.reshape(-1, 1)
                sens_run = sens

            else:
                y_run = y_prof[:, self.measured_ind]
                num_states = y_prof.shape[1]
                sens_run = self.select_sens(sens, num_states)

            y_data = self.y_data[ind].copy()
            if self.x_masks[ind] is not None:
                y_data[~self.x_masks[ind]] = y_run[~self.x_masks[ind]]

            resid_run = y_run - y_data

            # Store
            y_runs.append(y_run)
            resid_runs.append(resid_run)

            sens_by_y = reorder_sens(sens_run)
            weighted_sens = np.dot(sens_by_y, self.sigma_inv)

            weighted_sens = reorder_sens(weighted_sens,
                                         num_rows=len(self.x_model[ind]))

            sens_runs.append(weighted_sens)

        weighted_residuals = [np.dot(resid, self.sigma_inv)
                              for resid in resid_runs]

        if type(self.objfun_iter) is list:
            objfun_val = np.linalg.norm(np.concatenate(resid_runs))**2
            self.objfun_iter.append(objfun_val)

        residuals = self.optimize_flag * np.concatenate(resid_runs)

        if set_self:
            self.sens_runs = sens_runs
            self.y_runs = y_runs
            self.resid_runs = resid_runs

            self.residuals = residuals

        # Return objective
        if residual_vec:
            residual_out = np.concatenate([ar.T.ravel()
                                           for ar in weighted_residuals])
            return residual_out
        else:
            residual = 1/2 * residuals.T.dot(residuals)
            return residual

    def get_gradient(self, params, jac_matrix=False):
        if self.sens_runs is None:  # TODO: this is a hack to allow IPOPT
            self.get_objective(params)

        concat_sens = np.vstack(self.sens_runs)
        if not self.fit_spectra:

            if len(self.map_variable) == concat_sens.shape[1]:
                concat_sens = concat_sens[:, self.map_variable]

        self.sens = concat_sens
        jacobian = concat_sens

        if jac_matrix:
            return jacobian.T  # LM doesn't require (y - y_e)^T J
        else:
            gradient = jacobian.T.dot(self.residuals)  # 1D
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

            col_names = ['obj_fun'] + self.name_params
            self.paramest_df = pd.DataFrame(
                np.column_stack((self.objfun_iter, self.params_iter)),
                columns=col_names)

        # Model prediction with final parameters
        for ind in range(self.num_datasets):
            y_model = self.resid_runs[ind] + self.y_data[ind]
            self.y_model.append(y_model)

        covar_params = self.get_covariance()

        return opt_par, covar_params, info

    def get_covariance(self):
        jac = self.info_opt['jac']
        resid = self.info_opt['fun']

        hessian_approx = np.dot(jac, jac.T)

        dof = self.num_data_total - self.num_params
        mse = 1 / dof * np.dot(resid.T, resid)

        covar = mse * np.linalg.inv(hessian_approx)

        # Correlation matrix
        sigma = np.sqrt(covar.diagonal())
        d_matrix = np.diag(1/sigma)
        correlation = d_matrix.dot(covar).dot(d_matrix)

        self.covar_params = covar
        self.correl_params = correlation

        return covar

    def plot_data_model(self, **fig_kwargs):

        num_plots = self.num_datasets

        if 'ncols' not in fig_kwargs and 'nrows' not in fig_kwargs:
            num_cols = bool(num_plots // 2) + 1
            num_rows = num_plots // 2 + num_plots % 2

            fig_kwargs.update({'nrows': num_rows, 'ncols': num_cols})

        fig, axes = plt.subplots(**fig_kwargs)

        if num_plots == 1:
            axes = np.asarray(axes)[np.newaxis]

        ax_kwargs = {'mfc': 'None', 'ls': '', 'ms': 4}

        ax_flatten = axes.flatten()

        x_data = self.x_model
        y_data = self.y_data

        for ind in range(self.num_datasets):  # experiment loop
            mask_nan = np.isfinite(self.y_model[ind])
            y_model = self.y_model[ind]

            # Model prediction
            for col, mask in enumerate(mask_nan.T):
                ax_flatten[ind].plot(x_data[ind][mask], y_model[mask, col])

            lines = ax_flatten[ind].lines
            colors = [line.get_color() for line in lines]

            markers = cycle(['o', 's', '^', '*', 'P', 'X'])
            for color, y in zip(colors, y_data[ind].T):
                ax_flatten[ind].plot(x_data[ind], y, color=color,
                                     marker=next(markers),
                                     **ax_kwargs)

            # Edit
            ax_flatten[ind].spines['right'].set_visible(False)
            ax_flatten[ind].spines['top'].set_visible(False)

            # ax_flatten[ind].set_xlabel('$x$')
            ax_flatten[ind].set_ylabel(r'$\mathbf{y}$')

            ax_flatten[ind].xaxis.set_minor_locator(AutoMinorLocator(2))
            ax_flatten[ind].yaxis.set_minor_locator(AutoMinorLocator(2))

        if len(ax_flatten) > self.num_datasets:
            fig.delaxes(ax_flatten[-1])

        if len(axes) == 1:
            axes = axes[0]
            axes.set_xlabel('$x$')
        else:
            fig.text(0.5, 0, '$x$', ha='center')

        fig.tight_layout()

        return fig, axes

    def plot_data_model_sep(self, fig_size=None, fig_kwargs=None, dataset=0):

        xdata = self.x_model[dataset]

        ydata = self.y_data[dataset].T
        ymodel = self.y_model[dataset].T  # TODO: change this

        num_plots = ydata.shape[0]

        num_col = 2
        num_row = num_plots // num_col + num_plots % num_col

        num_row = 2
        num_col = num_plots // num_row + num_plots % num_row

        fig, axes = plt.subplots(num_row, num_col, figsize=fig_size)

        for ind in range(num_plots):  # for every state
            axes.flatten()[ind].plot(xdata, ymodel[ind])

            line = axes.flatten()[ind].lines[0]
            color = line.get_color()
            axes.flatten()[ind].plot(xdata, ydata[ind], 'o',
                                     color=color, mfc='None')

            axes.flatten()[ind].set_ylabel(self.name_states[ind])

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

    def plot_parity(self, fig_size=(4.5, 4.0), **fig_kwargs):
        if len(fig_kwargs) == 0:
            fig_kwargs['alpha'] = 0.70

        fig, axis = plt.subplots(figsize=fig_size)

        y_model = self.y_model

        if self.experim_names is None:
            experim_names = ['experiment {}'.format(ind + 1)
                             for ind in range(self.num_datasets)]
        else:
            experim_names = self.experim_names

        markers = cycle(['o', 's', '^', '*', 'P', 'X'])
        for ind, y_model in enumerate(y_model):
            y_data = self.y_data[ind]

            axis.scatter(y_model.T.flatten(), y_data.T.flatten(),
                         label=experim_names[ind], marker=next(markers),
                         **fig_kwargs)

        axis.set_xlabel('Model')
        axis.set_ylabel('Data')
        axis.legend(loc='best')

        plot_min, plot_max = axis.get_xlim()

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
                 global_analysis=True,
                 args_fun=None,
                 optimize_flags=None,
                 jac_fun=None, dx_finitediff=None,
                 measured_ind=None, weight_matrix=None,
                 name_params=None, name_states=None):

        super().__init__(func, param_seed, time_data, spectra,
                         args_fun, optimize_flags, jac_fun,
                         dx_finitediff, measured_ind, weight_matrix,
                         name_params, name_states)

        self.fit_spectra = True
        self.spectra = [data.T for data in self.y_data]
        self.len_spectra = [data.shape[0] for data in self.spectra]
        self.size_spectra = [data.size for data in self.spectra]

        self.spectra_tot = np.vstack(self.spectra)
        self.stdev_tot = np.concatenate(self.stdev_data)

        self.global_analysis = global_analysis

    def get_sens_projection(self, c_target, c_plus, sens_states, spectra_pred):
        eye = np.eye(c_target.shape[0])
        proj_orthogonal = eye - np.dot(c_target, c_plus)

        sens_pick = sens_states[self.map_variable][:, :, self.measured_ind]

        first_term = proj_orthogonal @ sens_pick @ c_plus
        second_term = first_term.transpose((0, 2, 1))
        sens_an = (first_term + second_term) @ spectra_pred

        n_par, n_times, n_lambda = sens_an.shape
        sens_proj = sens_an.T.reshape(n_lambda * n_times, n_par)

        return sens_proj

    def func_aux(self, params, x_vals, spectra, *args):
        states = self.function(params, x_vals, *args)

        _, epsilon, absorbance = mcr_spectra(
            states[:, self.measured_ind], spectra)

        return absorbance.T.ravel()

    def get_global_analysis(self, params,):
        c_runs = []
        sens_conc = []
        for ind in range(self.num_datasets):
            result = self.function(params, self.x_fit[ind],
                                   reorder=False,
                                   *self.args_fun[ind])

            if isinstance(result, tuple):
                conc_prof, sens_states = result
                sens_conc.append(sens_states)
            else:
                conc_prof = result

            conc_target = conc_prof[:, self.measured_ind]

            # MCR
            c_runs.append(conc_target)

        conc_tot = np.vstack(c_runs)

        # MCR
        conc_plus = np.linalg.pinv(conc_tot)
        absorptivity_pure = np.dot(conc_plus, self.spectra_tot)

        spectra_pred = np.dot(conc_tot, absorptivity_pure)

        if len(sens_conc) == 0:  # TODO: it won't work like this
            args_merged = [self.x_data[ind],
                           self.args_fun[ind], self.spectra[ind]]

            sens = numerical_jac_data(self.func_aux, params, args_merged,
                                      dx=self.dx_fd)[:, self.map_variable]
        else:
            sens_tot = np.concatenate(sens_conc, axis=1)
            sens = self.get_sens_projection(conc_tot, conc_plus,
                                            sens_tot, spectra_pred)

        residuals = (spectra_pred - self.spectra_tot).T.ravel() / \
            self.stdev_tot

        trim_y = np.cumsum(self.len_spectra)[:-1]
        trim_sens = np.cumsum(self.size_spectra)[:-1]

        y_runs = np.split(spectra_pred, trim_y, axis=0)
        y_runs = [array.T.ravel() for array in y_runs]

        sens_runs = np.split(sens, trim_sens, axis=0)
        resid_runs = np.split(residuals, trim_sens)

        self.epsilon_mcr = absorptivity_pure

        return y_runs, resid_runs, sens_runs

    def get_local_analysis(self, params):
        y_runs = []
        resid_runs = []
        sens_runs = []
        c_runs = []
        epsilon_mcr = []

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
            absorptivity_pure = np.dot(conc_plus, self.spectra[ind])
            spectra_pred = np.dot(conc_target, absorptivity_pure)

            epsilon_mcr.append(absorptivity_pure)

            if sens_states is None:
                args_merged = [self.x_data[ind],
                               self.args_fun[ind], self.spectra[ind]]

                sens = numerical_jac_data(self.func_aux, params, args_merged,
                                          dx=self.dx_fd)[:, self.map_variable]
            else:
                sens = self.get_sens_projection(conc_target, conc_plus,
                                                sens_states, spectra_pred)

            resid = (spectra_pred - self.spectra[ind]).T.ravel()
            resid_run = resid / self.stdev_data[ind]

            y_runs.append(spectra_pred.T.ravel())
            resid_runs.append(resid_run)
            sens_runs.append(sens)

        self.epsilon_mcr = epsilon_mcr

        return y_runs, resid_runs, sens_runs

    def get_objective(self, params, residual_vec=False):
        # Reconstruct parameter set with fixed and non-fixed indexes
        params = self.reconstruct_params(params)

        if type(self.params_iter) is list:
            self.params_iter.append(params)

        if self.global_analysis:
            y_runs, resid_runs, sens_runs = self.get_global_analysis(params)

        else:
            y_runs, resid_runs, sens_runs = self.get_local_analysis(params)

        self.y_runs = y_runs
        self.sens_runs = sens_runs
        self.resid_runs = resid_runs

        if type(self.objfun_iter) is list:
            objfun_val = np.linalg.norm(np.concatenate(self.resid_runs))**2
            self.objfun_iter.append(objfun_val)

        residuals = np.concatenate(resid_runs)
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
