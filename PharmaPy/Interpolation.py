#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 22:06:49 2020

@author: casas100
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


class NewtonInterpolation:
    def __init__(self, x_data, y_data):

        self.x_data = x_data
        self.y_data = y_data

        self.coeff = self.__getCoefficients()

    def __getCoefficients(self):
        """
        x: list or np array contanining x data points
        y: list or np array contanining y data points
        """

        n = len(self.x_data)
        x = self.x_data

        a = np.copy(self.y_data)

        if a.ndim == 1:
            for k in range(1, n):
                a[k:n] = (a[k:n] - a[k - 1]) / (x[k:n] - x[k - 1])

        else:
            for k in range(1, n):
                coeff = (a[k:n] - a[k - 1]).T / (x[k:n] - x[k - 1])
                a[k:n] = coeff.T

        return a

    def evalPolynomial(self, x):
        """
        x_data: data points at x
        y_data: data points at y
        x: evaluation point(s)
        """

        a = self.coeff
        n = len(self.x_data) - 1  # Degree of polynomial
        p = a[n]

        if isinstance(x, float) or isinstance(x, int):
            x_eval = x
        elif a.ndim == 1:
            x_eval = x
        else:
            x_eval = x[..., np.newaxis]

        for k in range(1, n + 1):
            p = a[n - k] + (x_eval - self.x_data[n - k])*p

        return p


class SplineInterpolation:
    def __init__(self, x_data, y_data):
        self.Spline = CubicSpline(x_data, y_data)

    def evalSpline(self, x):
        y_interp = self.Spline(x)
        return y_interp



if __name__ == '__main__':
    from scipy.interpolate import interp1d

    case = 2

    if case == 1:

        x_discr = np.arange(0, 5)
        y_discr = np.sin(x_discr)

        x_interp = [0.5]
        interp = NewtonInterpolation(x_discr, y_discr)
        y_interp = interp.evalPolynomial(x_interp)

        plots = False

        if plots:
            x_c = np.linspace(0, 5)
            y_c = np.sin(x_c)

            plt.plot(x_discr, y_discr, 'o', mfc='None')
            plt.plot(x_c, y_c)

            plt.plot(x_interp, interp, 's', mfc='None')

    elif case == 2:  # Saturation example
        sat = np.array(
                [0.2322061, 0.24866445, 0.25621458, 0.26201002, 0.26689287,
                 0.27117952, 0.27503542, 0.27856042, 0.28181078, 0.28464856])
        sat_two = 1.2 * sat + 0.05

        sat = np.column_stack((sat, sat_two))
        num_nodes = len(sat)
        step_grid = 0.06 / num_nodes
        z_sat = np.linspace(step_grid/2, 0.06 - step_grid/2, num_nodes)

        z_bounds = z_sat[[0, -1]]
        num_interp = 15
        nodes_interp = np.linspace(*z_bounds, num_interp)

        nodes_cont = np.linspace(*z_bounds, 100)

        scipy = False
        if scipy:
            sat_input_fit = interp1d(z_sat, sat)
            sat_interp_scipy = sat_input_fit(nodes_interp)
            sat_interp = sat_interp_scipy
            sat_cont = sat_input_fit(nodes_cont)
        else:
            interp = NewtonInterpolation(z_sat, sat)
            sat_interp_pharmapy = interp.evalPolynomial(nodes_interp)
            sat_interp = sat_interp_pharmapy

            sat_cont = interp.evalPolynomial(nodes_cont)

        plot = True
        if plot:
            fig, axis = plt.subplots()
            axis.plot(z_sat, sat, 'o', mfc='None')
            axis.plot(nodes_interp, sat_interp, 's', mfc='None')
            axis.plot(nodes_cont, sat_cont)
            axis.legend(('data', 'model'))
