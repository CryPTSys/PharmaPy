#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:01:59 2020

@author: dcasasor
"""

import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.animation import FuncAnimation
import numpy as np


def check_inputs(states, sep_states, width=4):
    if isinstance(states, list) or isinstance(states, tuple):
        list_states = []
        for state in states:
            if state.ndim == 1:
                list_states.append(np.atleast_2d(state).T)
            else:
                list_states.append(state)

        states = list_states

    else:

        if states.ndim == 1:
            states = [np.atleast_2d(states).T]
        else:
            states = [states]

    if sep_states:
        fig, axes = plt.subplots(len(states), 1,
                                 figsize=(width, width/1.6*len(states)))
        axes = np.atleast_1d(axes)
    else:
        fig, axes = plt.subplots()
        axes = [axes]

    return states, fig, axes


def anim_func(time, states, name_file=None, legend=None, names_y=None,
              title=None):
    """


    Parameters
    ----------
    time : array-like
        array containing the simulation time values
    states : list of array-like, array-like
        states to be animated. To plot several states, pass an array of each
        state in a list or tuple. Otherwise, pass the array corresponding to
        the state of interest
    name_file : str, optional
        Name of the video file. The default is None.
    legend : tuple of str, optional
        The default is None.
    names_y : tuple of str, optional
        Names of the states to be animated. The default is None.
    title : str, optional
        Title of the plot. The default is None.

    Returns
    -------
    animation : TYPE
        DESCRIPTION.
    fig : TYPE
        DESCRIPTION.
    axes : TYPE
        DESCRIPTION.

    """

    states, fig, axes = check_inputs(states, True)

    if legend is None:
        legend = [None] * len(states)
    else:
        legend = legend

    if name_file is None:
        name_file = 'anim'

    fig.suptitle(title)

    all_lines = []
    time_diff = time[-1] - time[0]
    xlim = (time[0] - time_diff*0.03, time[-1] + time_diff*0.03)
    for idx_state, state in enumerate(states):
        state_range = state.max() - state.min()
        ylim = (state.min() - state_range*0.03,
                state.max() + state_range*0.03)

        axes[idx_state].set_xlim(xlim)
        axes[idx_state].set_ylim(ylim)

        if names_y is not None:
            axes[idx_state].set_ylabel(names_y[idx_state])

        lines = []

        num_lines = state.shape[1]
        for i in range(num_lines):
            line_obj = axes[idx_state].plot([], [], lw=1.5)[0]
            lines.append(line_obj)

        all_lines.append(lines)

    time_tag = axes[0].text(
                1, 1.04, '$time = {:.1f}$ s'.format(time[0]),
                horizontalalignment='right',
                transform=axes[0].transAxes)

    axes[-1].set_xlabel('time (s)')

    def fun_anim(ind):
        for idx_state, state in enumerate(states):
            time_plot = time[:ind]
            states_plot = state[:ind]
            lines_state = all_lines[idx_state]

            for idx, col in enumerate(states_plot.T):  # index for each state
                lines_state[idx].set_data(time_plot, col)

                if legend[idx_state] is not None:
                    lines_state[idx].set_label(legend[idx_state][idx])

            time_tag.set_text('$time = {:.1f}$ s'.format(time[ind]))

            if legend[idx_state] is not None:
                axes[idx_state].legend(loc='best')

    fig.tight_layout()

    animation = FuncAnimation(fig, fun_anim,
                              frames=range(1, len(time)), repeat=True)

    writer = 'ffmpeg'
    suff = '.mp4'

    # writer = 'imagemagick'
    # suff = '.avi'

    animation.save(name_file + suff, writer=writer)

    return animation, fig, axes


def anim_multidim(time, indep_vble, data, filename=None, step_data=1,
                  title=None, xlabel=None, ylabel=None, invert_x=False,
                  time_unit=None, legend=None):
    """
    Animate states that depend on time and position. This typically arises when
    analyzing the outputs of a PDE model

    Parameters
    ----------
    time : array-like
        Time array from the simulator.
    indep_vble : numpy array
        array with the values of the spatial coordinate.
    data : numpy array
        independent variable to plot, with time advancing with rows, and space
        advancing with the columns..
    filename : str, optional
        Name of the output video file. The default is None.
    step_data : int, optional
        'data' array will be used as data[::step_data]. The default is 1.
    title : str, optional
        Figure title. The default is None.
    xlabel : str, optional
        DESCRIPTION. The default is None.
    ylabel : str, optional
        DESCRIPTION. The default is None.
    time_unit : str, optional
        If None, time unit is 's' (seconds). The default is None.
    legend : list of str, optional
        list of legends. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if filename is None:
        filename = 'anim'

    if time_unit is None:
        time_unit = 's'

    data, fig_anim, (ax_anim,) = check_inputs(data, sep_states=False)

    indep_diff = np.ptp(indep_vble)
    ax_anim.set_xlim(indep_vble.min() - 0.03*indep_diff,
                     indep_vble.max() + 0.03*indep_diff)
    fig_anim.suptitle(title)

    fig_anim.subplots_adjust(left=0, bottom=0, right=1, top=1,
                             wspace=None, hspace=None)

    data_min = np.vstack(data).min()
    data_max = np.vstack(data).max()

    data_diff = data_max - data_min

    ax_anim.set_ylim(data_min - 0.03*data_diff, data_max + data_diff*0.03)

    ax_anim.set_xlabel(xlabel)
    ax_anim.set_ylabel(ylabel)

    def func_data(ind):
        data_merged = []
        for idx_data, array in enumerate(data):
            data_merged.append(array[ind])

        if legend is not None:
            ax_anim.legend(legend)

        data_merged = np.column_stack(data_merged)
        return data_merged

    lines = ax_anim.plot(indep_vble, func_data(0))
    if invert_x:
        ax_anim.set_xlim(ax_anim.get_xlim()[::-1])

    time_tag = ax_anim.text(
        1, 1.04, '$time = {:.1f}$ {}'.format(time[0], time_unit),
        horizontalalignment='right',
        transform=ax_anim.transAxes)

    def func_anim(ind):
        f_vals = func_data(ind)
        for idx_line, line in enumerate(lines):
            line.set_ydata(f_vals[:, idx_line])

        # ax_anim.legend()
        fig_anim.tight_layout()

        time_tag.set_text('$t = {:.1f}$ {}'.format(time[ind], time_unit))

    frames = np.arange(0, len(time), step_data)
    animation = FuncAnimation(fig_anim, func_anim, frames=frames,
                              repeat=True)

    writer = 'ffmpeg'
    suff = '.mp4'

    animation.save(filename + suff, writer=writer)

    return animation, fig_anim, ax_anim
