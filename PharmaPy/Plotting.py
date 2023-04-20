# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:02:01 2022

@author: dcasasor
"""

import matplotlib.pyplot as plt
import numpy as np

from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.Commons import retrieve_pde_result


special = ('alpha', 'beta', 'gamma', 'phi', 'rho', 'epsilon', 'sigma', 'mu',
           'nu', 'psi', 'pi', '#')


def latexify_name(name, units=False):
    parts = name.split('/')

    out = []
    count = 0
    for part in parts:
        sep = None
        if '**' in part:
            segm = part.split('**')
            sep = '^'
        elif '_' in part:
            segm = part.split('_')
            sep = '_'
        else:
            segm = [part]

        for ind, s in enumerate(segm):
            if s in special:
                segm[ind] = '\\' + s

        if sep is None:
            if count > 0:
                part = part + '^{-1}'
            else:
                part = segm[0]
        else:
            inv = ''
            if count > 0:
                inv = '-'

            part = segm[0] + sep + '{' + inv + segm[1] + '}'

        out.append(part)
        count += 1

    if len(out) > 1:
        out = ' \ '.join(out)
    else:
        out = out[0]

    if units:
        out = '$\mathregular{' + out + '}$'
    else:
        out = '$' + out + '$'

    return out


def color_axis(ax, color):
    ax.spines['right'].set_color(color)
    ax.tick_params(axis='y', colors=color, which='both')
    ax.yaxis.label.set_color(color)


def get_indexes(names, picks):
    names = [a for a in names]
    out = []

    lower_names = [str(a).lower() if isinstance(a, str) else a for a in names]

    for pick in picks:
        if isinstance(pick, str):
            low_pick = pick.lower()
            if low_pick in lower_names:
                out.append(lower_names.index(low_pick))
            else:
                mess = "Name '%s' not in the set of compound names listed in the pure-component json file" % low_pick
                raise PharmaPyValueError(mess)

        elif isinstance(pick, (int, np.int32, np.int64)):
            out.append(pick)

    return out


def get_state_data(uo, *state_names):

    time = uo.timeProf
    di = {}
    for name in state_names:
        idx = None
        if isinstance(name, (tuple, list, range)):
            state, idx = name
        else:
            state = name

        y = getattr(uo, state + 'Prof')
        if idx is not None:
            y = y[:, idx]

        di[state] = y

    return time, di


def get_state_names(state_list):
    out = []
    for state in state_list:
        if isinstance(state, (list, tuple)):
            state = state[0]
        out.append(state)

    return out


def get_state_distrib(result, *state_names, **kwargs_retrieve):

    states = get_state_names(state_names)
    di = retrieve_pde_result(result, states=states, **kwargs_retrieve)
    out = {}
    for name in state_names:
        idx = None
        if isinstance(name, (tuple, list, range)):
            state, idx = name
            indexes = result.di_states[state]['index']
            idx = [indexes[i]
                   if isinstance(i, (int, np.int32, np.int64)) else i
                   for i in idx]
        else:
            state = name

        y = di[state]
        if idx is None:
            if isinstance(y, dict):
                y = list(y.values())
        else:
            y = [y[i] for i in idx]

        out[state] = y

    return out


def get_states_result(result, *state_names):
    time = result.time
    states_fstates = result.di_states | result.di_fstates

    out = {}
    for key in state_names:
        idx = None
        if isinstance(key, (list, tuple, range)):
            state, idx = key
            indexes = states_fstates[state]['index']
            idx = get_indexes(indexes, idx)
        else:
            state = key

        y = getattr(result, state)

        if idx is not None:
            y = y[:, idx]

        out[state] = y

    return time, out


def name_yaxes(ax, states_fstates, names, ylabels, legend):
    for ind, name in enumerate(names):
        axis = ax[ind]
        if ylabels is None:
            ylabel = names[ind]
        else:
            ylabel = latexify_name(ylabels[ind])

        units = states_fstates[name].get('units', '')
        if len(units) > 0:
            unit_name = latexify_name(units, units=True)
            ylabel = ylabel + ' (' + unit_name + ')'

        axis.set_ylabel(ylabel)


def set_legend(ax, states_fstates, names, state_names, legend):
    for ind, name in enumerate(names):
        index_y = states_fstates[name].get('index', False)
        if index_y and legend:
            if isinstance(state_names[ind], (tuple, list)):
                picks = state_names[ind][1]
                picks = get_indexes(index_y, picks)

                index_y = [index_y[i] for i in picks]

            ax[ind].legend(index_y, loc='best')


def plot_function(uo, state_names, axes=None, fig_map=None, ylabels=None,
                  include_units=True, **fig_kwargs):
    time, data = get_states_result(uo.result, *state_names)

    if fig_map is None:
        fig_map = range(len(data))

    if axes is None:
        fig, ax_orig = plt.subplots(**fig_kwargs)
    else:
        ax_orig = axes

    if isinstance(ax_orig, np.ndarray):
        axes = ax_orig.flatten()
    else:
        axes = (ax_orig, )

    count = 0
    linestyles = ('-', '--', '-.', ':')
    colors = plt.cm.tab10

    names = list(data.keys())
    states_and_fstates = {**uo.states_di, **uo.fstates_di}

    for ind, idx in enumerate(fig_map):
        name = names[ind]
        y = data[name]
        twin = False

        # index_y = False
        index_y = states_and_fstates[name].get('index', False)

        if isinstance(state_names[ind], (tuple, list, range)):
            y_ind = state_names[ind][1]
            y_ind = get_indexes(index_y, y_ind)

            index_y = [index_y[a] for a in y_ind]

        if len(axes[idx].lines) > 0:
            ax = axes[idx].twinx()
            count += len(axes[idx].lines)
            twin = True
        else:
            ax = axes[idx]

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for sp, row in enumerate(y.T):
            ax.plot(time, row, color=colors(count),
                    linestyle=linestyles[count % len(linestyles)])

            if twin:
                color_axis(ax, colors(count))

            count += 1

        if ylabels is None:
            ylabel = name
        else:
            ylabel = latexify_name(ylabels[ind])

        units = states_and_fstates[name].get('units', '')
        if len(units) > 0:
            unit_name = latexify_name(states_and_fstates[name]['units'],
                                      units=True)
            ylabel = ylabel + ' (' + unit_name + ')'

        if index_y:
            ax.legend(index_y, loc='best')

        ax.set_ylabel(ylabel)

        count = 0

    if len(axes) == 1:
        axes = axes[0]

    # for ax in axes:
    #     if len(ax.lines) == 0:
    #         ax.remove()

    if 'fig' in locals():
        return fig, ax_orig
    else:
        return ax_orig


def plot_distrib(uo, state_names, x_name, axes=None, times=None, x_vals=None,
                 cm_names=None, ylabels=None, legend=True, **fig_kwargs):
    if times is None and x_vals is None:
        raise ValueError("Both 'times' and 'x_vals' arguments are None. "
                         "Please specify one of them")

    elif not isinstance(x_vals, (tuple, list)):
        x_vals = (x_vals, )

    if cm_names is None:
        cm_names = ['Blues', 'Oranges', 'Greens',  'Reds', 'Purples', ]
    elif isinstance(cm_names, str):
        cm_names = [cm_names]

    cm = [getattr(plt.cm, cm_name) for cm_name in cm_names]

    if axes is None:
        fig, ax = plt.subplots(**fig_kwargs)
    else:
        ax = axes

    if not isinstance(ax, np.ndarray):
        ax = [ax]
    else:
        ax = ax.flatten()

    states_and_fstates = {**uo.states_di, **uo.fstates_di}

    if times is not None:
        if len(times) == 1:
            colors = [[None]] * len(cm)
        else:
            ls = np.linspace(0.2, 0.8, len(times))
            colors = [cmap(ls) for cmap in cm]

        y = get_state_distrib(uo.result, *state_names, time=times,
                              x_name=x_name)

        names = list(y.keys())
        x_vals = getattr(uo.result, x_name)

        for t, time in enumerate(times):
            for ind, name in enumerate(names):
                axis = ax[ind]
                y_plot = y[name]

                if isinstance(y_plot, list):
                    for st, ar in enumerate(y_plot):
                        ind_cm = st % len(cm)
                        axis.plot(x_vals, ar[t], color=colors[ind_cm][t])
                else:
                    axis.plot(x_vals, y_plot[t], color=colors[0][t])

        name_yaxes(ax, states_and_fstates, names, ylabels, legend)
        set_legend(ax, states_and_fstates, names, state_names, legend)

        for axis in ax:
            if len(axis.lines) == 0:
                axis.remove()

        fig.text(0.5, 0, x_name)

        if len(ax) == 1:
            ax = ax[0]

    else:
        if len(x_vals) == 1:
            colors = [[None]] * len(cm)
        else:
            ls = np.linspace(0.2, 0.8, len(x_vals))
            colors = [cmap(ls) for cmap in cm]

        y_di = get_state_distrib(uo.result, *state_names, x=x_vals,
                                 x_name=x_name)

        names = list(y_di.keys())

        time = uo.result.time
        for ct, x in enumerate(x_vals):
            for ind, (state, y) in enumerate(y_di.items()):
                if isinstance(y, list):
                    for st, ar in enumerate(y):
                        ind_cm = st % len(cm)
                        ax[ind].plot(time, ar[:, ct], color=colors[ind_cm][ct])
                else:
                    ax[ind].plot(time, y[:, ct], color=colors[0][ct])

        name_yaxes(ax, states_and_fstates, names, ylabels, legend)
        set_legend(ax, states_and_fstates, names, state_names, legend)

    if 'fig' in locals():
        return fig, ax
    else:
        return ax
