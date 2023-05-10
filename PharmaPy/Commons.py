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

from assimulo.exception import TerminateSimulation

linestyles = cycle(['-', '--', '-.', ':'])
eps = np.finfo(float).eps


def get_permutation_indexes(ref_list, short_list):
    """
    Permutate indexes of short list with respect to the order in which they
    appear in the reference list

    Parameters
    ----------
    ref_list : list
        reference list.
    short_list : list
        analyzed list. It must be a subset of ref_list.

    Returns
    -------
    indexes : list
        list of permutation indexes.

    """
    indexes = [short_list.index(val) for val in ref_list if val in short_list]

    return indexes


def flatten_states(state_list):
    name_states = list(state_list[0].keys())

    if len(state_list) == 1:
        return state_list[0]
    else:
        out = {}
        for name in name_states:
            ar = [di[name] if ind == 0 else di[name][1:]
                  for ind, di in enumerate(state_list)]
            if ar[0].ndim == 1:
                out[name] = np.concatenate(ar)
            else:
                out[name] = np.vstack(ar)

        return out


def unpack_discretized(states, num_states, name_states, indexes=None,
                       state_map=None, inputs=None):

    acum_len = np.cumsum(num_states)[:-1]

    if states.ndim == 1:
        states_reord = states.reshape(-1, sum(num_states))

        states_split = np.split(states_reord, acum_len, axis=1)
        states_split = [a[:, 0] if a.shape[1] == 1 else a
                        for a in states_split]

    elif states.ndim > 1:
        dim_tot = sum(num_states)
        num_fv = states.shape[1] // dim_tot
        num_times = states.shape[0]

        nums = list(num_states) * num_fv
        acum_len = np.cumsum(nums)[:-1]

        states_fv = np.split(states, acum_len, axis=1)

        count_states = len(name_states)
        states_split = []
        for idx_state, name in enumerate(name_states):
            state = np.vstack(states_fv[idx_state::count_states])

            di_key = indexes[name]

            if di_key is None:
                state_data = state.reshape(-1, num_times).T

                if inputs is not None and name in inputs:
                    state_data = np.column_stack((inputs[name], state_data))
            else:
                state_data = {}
                for idx_col in range(state.shape[1]):
                    di_data = state[:, idx_col].reshape(-1, num_times).T

                    if inputs is not None and name in inputs:
                        inpt = inputs[name]
                        di_data = np.column_stack((inpt[:, idx_col], di_data))

                    state_data[di_key[idx_col]] = di_data

            states_split.append(state_data)

    dict_states = dict(zip(name_states, states_split))

    return dict_states


def unpack_states(states, num_states, name_states, state_map=None):
    acum_len = np.cumsum(num_states)[:-1]

    if states.ndim == 1:
        states_split = np.split(states, acum_len)
        states_split = [a[0] if len(a) == 1 else a for a in states_split]
    elif states.ndim > 1:
        states_split = np.split(states, acum_len, axis=1)
        states_split = [a[:, 0] if a.shape[1] == 1 else a
                        for a in states_split]

    if state_map is not None:
        states_split = [array for ind, array in enumerate(states_split)
                        if state_map[ind]]

        name_states = [name for ind, name in enumerate(name_states)
                       if state_map[ind]]

    dict_states = dict(zip(name_states, states_split))

    return dict_states


def retrieve_pde_result(data, x_name, states=None, time=None, x=None,
                        idx_time=None, idx_vol=None):

    if isinstance(data, dict):
        di = data
    elif data.__class__.__name__ == 'DynamicResult':
        di = data.__dict__

    out = {}

    if idx_time is None:
        if time is None:
            idx_time = np.arange(len(di['time']))
        elif isinstance(time, (list, tuple, np.ndarray)):
            # TODO: Should we interpolate instead?
            idx_time = [np.argmin(abs(t - di['time'])) for t in time]
        else:
            idx_time = np.argmin(abs(time - di['time']))
            out['time'] = di['time'][idx_time]

    if idx_vol is None:
        if x is None:
            idx_vol = np.arange(len(di[x_name]))
        elif isinstance(x, (list, tuple, np.ndarray)):
            idx_vol = [np.argmin(abs(val - di[x_name])) for val in x]
            out[x_name] = di[x_name][idx_vol]
        else:
            idx_vol = np.argmin(abs(x - di[x_name]))
            out[x_name] = di[x_name][idx_vol]

    di_filtered = {key: di[key] for key in di if key != 'time' and key != x_name}

    if states is None:
        states = list(di_filtered.keys())

    for key in states:
        val = di_filtered[key]
        if isinstance(val, dict):
            out[key] = retrieve_pde_result(val, x_name, idx_time=idx_time,
                                           idx_vol=idx_vol)
        elif isinstance(val, np.ndarray):
            # if x_name in di['di_states'][key]['depends_on']:
            out[key] = val[idx_time][:, idx_vol]

    return out


def complete_dict_states(time, di, target_keys, phase, controls,
                         u_inputs=None, num_discr=1):
    for key in target_keys:
        if key not in di:
            if key in controls.keys():
                control = controls[key]
                di[key] = control['fun'](time,
                                         *control['args'],
                                         **control['kwargs'])
            else:
                val = getattr(phase, key, None)

                time_flag = isinstance(time, (list, np.ndarray)) and len(time) > 1

                if num_discr > 1:
                    if time_flag:
                        val = val * np.ones((len(time), num_discr))
                    else:
                        val = val * np.ones(num_discr)

                elif time_flag:
                    val = val * np.ones_like(time)

                di[key] = val

        if u_inputs is not None:
            if di[key].ndim == 1:
                di[key] = np.hstack((u_inputs[key], di[key]))
            else:
                di[key] = np.vstack((u_inputs[key], di[key]))

    return di


def check_steady_state(time, states, sdot, tau, num_tau=1, time_stop=None,
                       threshold=1e-5, norm_type=None):

    if not isinstance(threshold, (tuple, list)):
        threshold = [threshold] * len(sdot)

    sdot_flags = []

    norms = []
    for val in sdot.values():
        norms.append(np.linalg.norm(val, ord=norm_type))

    if time_stop is None:
        time_limit = num_tau * tau
    else:
        time_limit = time_stop

    time_flag = time_limit < time
    sdot_flags = [nor < lim for nor, lim in zip(norms, threshold)]

    flags = [time_flag] + sdot_flags

    flag = not all(flags)

    return float(flag)


def eval_state_events(time, states, switches,
                      states_dim, name_states, state_event_list,
                      sdot=None, discretized_model=False, state_map=None):
    events = []

    if discretized_model:
        unpack_fn = globals()['unpack_discretized']
    else:
        unpack_fn = globals()['unpack_states']

    dict_states = unpack_fn(states, states_dim, name_states)

    if sdot is not None:
        dict_sdot = unpack_fn(sdot, states_dim, name_states,
                              state_map=state_map)

    if any(switches):

        for di in state_event_list:
            if 'callable' in di.keys():
                kwargs_callable = di.get('kwargs', {})
                event_flag = di['callable'](time, dict_states, dict_sdot,
                                            **kwargs_callable)
            else:
                state_name = di['state_name']
                ref_value = di['value']

                state_idx = di.get('state_idx', None)

                if state_idx is None:
                    checked_value = dict_states[state_name]
                else:
                    checked_value = dict_states[state_name][state_idx]

                event_flag = ref_value - checked_value

            events.append(event_flag)

    events = np.hstack(events)

    return events


def handle_events(solver, event_info, state_event_list, any_event=True):
    # if any_event:
    event_markers = event_info[0]

    flags = []

    dim_events = [event.get('num_conditions', 1) for event in state_event_list]

    idx_state = [[ind] * num for ind, num in enumerate(dim_events)]
    idx_state = np.hstack(idx_state)

    for ind, val in enumerate(event_markers):
        direction = state_event_list[idx_state[ind]].get('direction')
        terminate = False

        if val:
            if direction is None:
                terminate = True
            elif direction == val:
                terminate = True

        flags.append(terminate)

    if any_event:
        if any(flags):
            idx_true = [ind for (ind, flag) in enumerate(flags) if flag]

            for idx in idx_true:
                id_event = state_event_list[idx].get('event_name')
                if flags[idx]:
                    if id_event is None:
                        print('State event %i was reached' % (idx + 1))
                    else:
                        print("State event '%s' was reached" % id_event)

                    raise TerminateSimulation

    else:
        if all(flags):
            raise TerminateSimulation


def high_resolution_fvm(y, boundary_cond, limiter_type='Van Leer',
                        both=False):

    # Ghost cells -1, 0 and N + 1 (see LeVeque 2002, Chapter 9)
    y_extrap = 2*y[-1] - y[-2]
    y_aug = np.concatenate(([boundary_cond]*2, y, [y_extrap]))

    y_diff = np.diff(y_aug, axis=0)

    theta = (y_diff[:-1]) / (y_diff[1:] + eps)

    if limiter_type == 'Van Leer':
        limiter = (np.abs(theta) + theta) / (1 + np.abs(theta))
    else:  # TODO: include more limiters
        pass

    fluxes = y_aug[1:-1] + 0.5 * y_diff[1:] * limiter

    return fluxes


def upwind_fvm(y, boundary_cond):
    y_aug = np.concatenate(([boundary_cond], y))

    y_diff = np.diff(y_aug)

    return y_diff


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


def build_pw_lin(time_vals=None, time_lengths=None, y_vals=None, y_ramps=None,
                 t_init=0.0, y_init=None):

    if y_ramps is not None and y_init is None:
        raise ValueError('If specifying ramp functions, you need to pass an '
                         'initial state value for y_init')

    if time_vals is None and time_lengths is None:
        raise ValueError('time_vals or time_lengths must be specified to '
                         'utilize piecewise linear profiles')

    if y_vals is None and y_ramps is None:
        raise ValueError('y_vals or y_ramps must be specified to utilize '
                         'piecewise linear profiles')

    num_segments = 0
    if time_vals is None:
        num_segments = len(time_lengths)
    else:
        num_segments = len(time_vals) - 1

    pw_logic_exprs = []
    pw_fncs = []
    y_prev = None
    y_end = None

    for j in range(num_segments):
        if time_vals is not None:
            def fun_logic(x, j=j): return \
                (time_vals[j] <= x) * (x < time_vals[j + 1])

            if y_vals is not None:
                def function(x, j=j): return \
                    (x - time_vals[j])/(time_vals[j + 1] - time_vals[j]) *\
                    (y_vals[j + 1] - y_vals[j]) + y_vals[j]

                y_end = y_vals[j + 1]
            else:
                if j == 0:
                    y_prev = y_init

                def function(x, j=j, y_prev=y_prev): return \
                    (x - time_vals[j]) * y_ramps[j] + y_prev

                y_prev += y_ramps[j] * (time_vals[j + 1] - time_vals[j])
                y_end = y_prev

            pw_logic_exprs.append(fun_logic)
            pw_fncs.append(function)

            if j == (num_segments - 1):
                pw_logic_exprs.append(lambda x, j=j: (x >= time_vals[j + 1]))
                pw_fncs.append(lambda x, j=j, y_end=y_end: y_end)
        else:
            def fun_logic(x, j=j): return \
                (sum(time_lengths[0:j]) <= x)*(x < sum(time_lengths[0:j + 1]))

            if y_vals is not None:
                def function(x, j=j): return \
                    (x - sum(time_lengths[0:j])) / (time_lengths[j]) * \
                    (y_vals[j + 1] - y_vals[j]) + y_vals[j]

                y_end = y_vals[j + 1]
            else:
                if j == 0:
                    y_prev = y_init

                def function(x, j=j, y_prev=y_prev): return \
                    (x - sum(time_lengths[0:j])) * y_ramps[j] + y_prev

                y_prev += y_ramps[j] * (time_lengths[j])
                y_end = y_prev

            pw_logic_exprs.append(fun_logic)
            pw_fncs.append(function)

            if j == (num_segments - 1):
                pw_logic_exprs.append(lambda x, j=j:
                                      (x >= sum(time_lengths[0:j + 1])))
                pw_fncs.append(lambda x, j=j, y_end=y_end: y_end)

    return pw_logic_exprs, pw_fncs


def temp_pw_lin(time, pw_exprs=None, pw_fncs=None, t_zero=0):
    pw_exprs_vals = []

    for ind, val in enumerate(pw_exprs):
        if ind < (len(pw_exprs)):
            pw_exprs_vals.append(pw_exprs[ind](time))

    temp_t = np.piecewise(time, pw_exprs_vals, pw_fncs)

    return temp_t


def plot_sens(time_prof, sensit, fig_size=None, name_states=None,
              name_params=None, mode='per_parameter', black_white=False,
              time_div=1):

    num_plots = len(sensit)
    num_cols = bool(num_plots // 2) + 1
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
        legend = [r'$' + name + r'$' for name in name_params]
        # legend = [r'${}$'.format(name_params[ind])
        #           for ind in range(num_params)]
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


def reorder_sens(sens, separate_sens=False, num_rows=None):
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

    if isinstance(sens, (tuple, list)):
        num_rows = sens[0].shape[0]
    elif isinstance(sens, np.ndarray):
        if num_rows is None:
            raise ValueError("'num_times' argument has to be passed if 'sens' "
                             "is a numpy array")

    big_sens = np.vstack(sens)

    ordered_sens = []
    for col in big_sens.T:
        reordered = col.reshape(-1, num_rows).T
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
