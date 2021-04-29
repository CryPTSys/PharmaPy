# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:22:20 2020

@author: dcasasor
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# from autograd import numpy as np
import numpy as np
from scipy.integrate import simps
from itertools import cycle

linestyles = cycle(['-', '--', '-.', ':'])


def geom_series(start, stop, num, rate_out=False):
    rate = (stop/start)**(1/(num - 1))

    series = start*rate**np.arange(0, num)

    if rate_out:
        return rate
    else:
        return series


def geom_counts(x_target, x_max, *args_geom):
    rate = geom_series(*args_geom, rate_out=True)

    n_target = np.log(x_target/x_max) / np.log(rate)
    n_target = np.floor(n_target) + args_geom[-1]

    return n_target


def series_erfc(x, num=64):
    if isinstance(x, np.ndarray):
        x = x[..., np.newaxis]

    ind = np.arange(1, num + 1, dtype=np.uint64)
    odds = np.arange(1, 2*num, 2, dtype=np.uint64)

    series = (-1)**ind * np.cumprod(odds) / (2 * x**2)**ind
    result = 1 / np.sqrt(np.pi) * (1 + series.T.sum(axis=0))

    return result


def temp_ufun(time, temp_zero, rate, t_zero=0):
    temp_t = temp_zero + rate*(time - t_zero)

    return temp_t


def vol_ufun(time, vol_zero, rate):
    vol_t = vol_zero + rate*time
    return vol_t


def plot_sens(time_prof, sensit, fig_size=None, name_states=None,
              name_params=None, mode='per_parameter', black_white=False,
              time_div=1):

    num_plots = len(sensit)
    num_cols = bool(num_plots // 2) + 2
    num_rows = num_plots // num_cols + (num_plots % num_cols > 0)

    time_prof *= 1 / time_div

    if fig_size is None:
        fig_size = (6, 6/num_cols/1.4 * num_rows)

    fig_sens, axes_sens = plt.subplots(num_rows, num_cols,
                                       figsize=fig_size)

    if num_plots == 1:
        axes_sens = np.asarray(axes_sens)[np.newaxis]

    count = 0
    for ax, sens in zip(axes_sens.flatten(), sensit):
        if black_white:
            for col in sens.T:
                ax.plot(time_prof, col,
                        # 'k',
                        linestyle=next(linestyles))
        else:
            ax.plot(time_prof, sens)

        # ax.set_xscale('log')

        if time_div == 1:
            ax.set_xlabel(r'time (s)')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # ax.set_ylabel(r'$\frac{ds_i}{d \theta_{%i}}$' % count)

        if mode == 'per_parameter':
            ax.set_ylabel(r'$\dfrac{\partial C_i}{\partial %s}$'
                          % name_params[count])
        elif mode == 'per_state':
            ax.set_ylabel(r'$\dfrac{\partial %s}{\partial \theta_j}$'
                          % name_states[count])

        count += 1

    num_states = len(name_states)
    num_params = len(name_params)

    num_subplots = len(axes_sens.ravel())
    if num_subplots > num_plots:
        [fig_sens.delaxes(ax) for ax in axes_sens.flatten()[num_plots:]]

    if mode == 'per_parameter':
        legend = [r'$' + name + r'$' for name in name_states]
        if len(legend) > 4:
            ncols = num_states // 4 + 1
        else:
            ncols = 1
    elif mode == 'per_state':
        legend = [r'${}$'.format(name_params[ind])
                  for ind in range(num_params)]
        if len(legend) > 4:
            ncols = num_params // 4 + 1
        else:
            ncols = 1

    axes_sens.flatten()[0].legend(legend, ncol=ncols, handlelength=2)
    fig_sens.tight_layout()

    return fig_sens, axes_sens


def integration(states, time):
    if states.ndim == 1:
        states = states[..., np.newaxis]

    integral = np.zeros(states.shape[1])
    for ind, row in enumerate(states.T):
        integral[ind] = simps(row, time)

    return integral


def reorder_sens(sens, separate_sens=False):
    """
    Reorder sensitivities into submatrices according to the following pattern,
    for ns states, np parameters and nt time samples:

        ds_1/dp_1 (t_1) ... ds_1/dp_np(t_1)
        .
        .
        .
        ds_1/dp_1 (t_nt) ... ds_1/dp_np(t_nt)


        ds_2/dp_1 (t_1) ... ds_2/dp_np(t_1)
        .
        .
        .
        ds_2/dp_1 (t_nt) ... ds_2/dp_np(t_nt)

        ...

        ds_ns/dp_1 (t_1) ... ds_ns/dp_np(t_1)
        .
        .
        .
        ds_ns/dp_1 (t_nt) ... ds_ns/dp_np(t_nt)

    Originally, sensitivities are grouped by SUNDIALS as
    (example for the first submatrix):

        ds_1/dp_1 (t_1) ... ds_ns/dp_1(t_1)
        .
        .
        .
        ds_1/dp_1 (t_nt) ... ds_ns/dp_1(t_nt)



    Parameters
    ----------
    sens : list of arrays
        sensitivities as returned by SUNDIALS.
    separate_sens : bool, optional
        If True, submatrices are returned as a list of arrays. If False,
        submatrices are stacked one below another. The default is False.

    Returns
    -------
    sens_concat : list or array
        reordered sensitivity.

    """

    num_times, num_states = sens[0].shape

    big_sens = np.vstack(sens)

    ordered_sens = []
    for col in big_sens.T:
        reordered = col.reshape(-1, num_times).T
        ordered_sens.append(reordered)

    if separate_sens:
        return ordered_sens
    else:
        sens_concat = np.vstack(ordered_sens)
        return sens_concat


def mid_fn(x):
    mid = np.sum(x) - np.min(x) - np.max(x)
    return mid


def trapezoidal_rule(x_vals, y_vals):
    x_term = np.diff(x_vals)
    y_term = (y_vals[1:] + y_vals[:-1]) / 2

    integral = np.dot(x_term, y_term)

    return integral


def reorder_pde_outputs(state_array, num_fv, size_states, name_states=None):
    states_splitted = np.split(state_array, num_fv, axis=1)
    states_per_fv = []

    num_times = state_array.shape[0]

    for array in states_splitted:
        list_temp = []
        count = 0
        for size in size_states:
            state_indiv = array[:, count:size + count]
            list_temp.append(state_indiv)

            count += size

        states_per_fv.append(list_temp)

    individual_states = []
    states_stacked = np.vstack(states_splitted)

    count = 0
    for size in size_states:
        state_indiv = states_stacked[:, count:size + count]
        state_indiv = state_indiv.T.reshape(-1, num_times).T

        if size > 1:
            state_indiv = np.split(state_indiv, size, axis=1)

        individual_states.append(state_indiv)

        count += size

    if name_states:
        individual_states = {
            key: value for key, value in zip(name_states, individual_states)}

    return states_per_fv, individual_states
