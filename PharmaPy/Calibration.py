#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:29:05 2020

@author: dcasasor
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


class PCR_calibration:
    def __init__(self, data, num_comp=None, standardize=True, snv=False,
                 y_name=None, y_suffixes=None):

        self.data = data
        self.standardize = standardize
        self.snv = snv

        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)

        if snv:
            self.data_centered = self.__center_data(data)
        else:
            self.data_centered = self.__center_data(data, data_mean, data_std)
            # self.data_centered = self.__center_data(data, None, None)

        self.data_mean = data_mean
        self.data_std = data_std

        (self.projections, self.explained_variance,
         self.svd_dict) = self.__get_projections()

        if num_comp is None:
            self.num_comp = len(self.svd_dict['sv'])
        else:
            self.num_comp = num_comp

        if y_name is None:
            y_name = 'y_'

        self.y_name = y_name
        self.y_suffixes = y_suffixes

    def __center_data(self, data=None, mean=None, std=None):
        if mean is None and std is None:
            mean = data.mean(axis=0)
            std = data.std(axis=0)

        data_centered = data - mean

        if self.snv or self.standardize:
            data_centered *= 1 / std

        return data_centered

    def __get_projections(self, data=None, n_comp=None):

        if data is None:
            data = self.data_centered

        # Perform SVD
        u_m, sv, v_nt = np.linalg.svd(data)
        v_n = v_nt.T

        # Percent of explained variance
        explained_var = sv**2 / (sv**2).sum() * 100

        # Store SVD in a dict
        v_trunc = v_nt[:len(sv)].T

        svd_dict = {'U': u_m, 'sv': sv, 'V': v_n, 'V_trunc': v_trunc}

        # Projections
        projections = np.dot(data, v_n)

        if n_comp is not None:
            projections = projections[:, :n_comp]

        return projections, explained_var, svd_dict

    def plot_projections(self, fig_size=None, num_comp=None):

        if num_comp is None:
            num_comp = self.num_comp

        comb = itertools.combinations(range(num_comp), 2)

        combs = [item for item in comb]
        num_plots = len(combs)

        if num_plots == 1:
            fig, axes = plt.subplots(figsize=fig_size)
            axes = np.atleast_1d(axes)
        else:
            ncols = 2
            nrows = num_plots // ncols + num_plots % ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)

        axes_flat = axes.flatten()
        num_axes = len(axes_flat)
        for ind in range(num_plots):
            axis = axes_flat[ind]
            pc_one, pc_two = combs[ind]

            my_map = plt.get_cmap('Reds')
            data_plot = self.projections[:, combs[ind]].T
            axis.scatter(data_plot[0], data_plot[1],
                         # 'o', mfc='None',
                         s=30/(num_axes/2), c=range(data_plot.shape[1]),
                         cmap=my_map,
                         marker='o', edgecolor='k')

            axis.text(0.5, -0.08, 'PC%i' % (pc_one + 1),
                      transform=axis.transAxes, ha='center')

            axis.text(-0.08, 0.5, 'PC%i' % (pc_two + 1), rotation=90,
                      transform=axis.transAxes, va='center')

            axis.spines['bottom'].set_position('zero')
            axis.spines['left'].set_position('zero')

            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)

        if num_plots < num_axes:
            fig.delaxes(axes_flat[-1])

        fig.tight_layout()

        return fig, axes

    def get_regression(self, y_data, num_comp=None, update_instance=True):

        if self.y_suffixes is None:
            self.y_suffixes = ['%i' % num for num in range(1, len(y_data))]

        self.y_labels = [r'$' + self.y_name + ('{%s}' % suffix) + '$'
                         for suffix in self.y_suffixes]

        if num_comp is None:
            num_comp = self.num_comp

        # scores = self.projections[:, :num_comp]
        scores = self.projections[:, :num_comp]

        scores_inv = np.linalg.pinv(scores)  # pseudoinverse of scores

        # Regression coefficients w.r.t. principal components
        if y_data.ndim == 1:
            y_data = y_data[..., np.newaxis]

        y_means = y_data.mean(axis=0)
        y_center = y_data - y_means
        q_coeff = np.dot(scores_inv, y_center)

        regression_coeff = q_coeff

        self.y_data = y_data
        self.y_center = y_center
        self.y_means = y_means

        if update_instance:
            self.regression_coeff = regression_coeff
            self.num_comp = num_comp

            y_pred = self.predict(self.data)
            residuals = y_data - y_pred
            self.residuals = residuals

        # # Regression coefficients w.r.t. original X
        # regression_coeff = np.dot(self.svd_dict['V'][:, :num_comp],
        #                           q_coeff)


        # if update_instance:
        #     self.regression_coeff = regression_coeff

        #     y_pred = self.predict(self.data)
        #     residuals = y_data - y_pred
        #     self.residuals = residuals

        return regression_coeff

    def predict(self, inputs, num_comp=None, regression_coeff=None,
                full_output=False):
        inputs = np.atleast_2d(inputs)

        if self.snv:
            inputs_centered = self.__center_data(inputs)
        else:
            inputs_centered = self.__center_data(inputs)

        if regression_coeff is None:
            coeff = self.regression_coeff
        else:
            coeff = regression_coeff

        if num_comp is None:
            num_comp = self.num_comp

        p_matrix = self.svd_dict['V'][:, :num_comp]
        new_projections = np.dot(inputs_centered, p_matrix)

        resid_x = inputs_centered[0] - np.dot(new_projections[0],
                                              p_matrix.T)

        SPE_x = np.dot(resid_x, resid_x)
        print(SPE_x)

        # new_projections, _, di = self.__get_projections(inputs_centered,
        #                                                 num_comp)


        response = np.dot(new_projections, coeff) + self.y_means

        num_data = self.y_data.size

        if full_output:
            resid = response - self.y_data
            mse = 1 / num_data * np.dot(resid.T, resid)
            info_out = {'x_projected': new_projections, 'y_pred': response,
                        'MSE': mse}

            return info_out
        else:

            return response

    def evaluate_mse(self, num_comp=None):

        if num_comp is None:
            pc_counter = range(len(self.svd_dict['sv']))
        else:
            pc_counter = range(num_comp)

        mse = []
        residuals = []

        n_data = np.prod(self.y_data.shape)
        for n_component in pc_counter:
            coeff = self.get_regression(self.y_data, n_component + 1,
                                        update_instance=False)

            pred = self.predict(self.data, regression_coeff=coeff,
                                num_comp=n_component + 1)

            resid = self.y_data - pred

            mse_val = 1 / n_data * (resid**2).sum()

            mse.append(mse_val)
            residuals.append(resid)

        return mse, residuals

    def plot_parity(self, figsize=None):
        if figsize is None:
            figsize = (4, 3.5)
        fig, axis = plt.subplots(figsize=figsize)

        markers = ['o', 's', 'd', '*']

        y_pred = self.y_data - self.residuals

        minim = (np.minimum(self.y_data, y_pred)).min()
        maxim = (np.maximum(self.y_data, y_pred)).max()

        range_vals = maxim - minim

        left_bottom = [minim - range_vals*0.01]*2
        right_top = [maxim + range_vals*0.01]*2

        axis.plot(*zip(left_bottom, right_top), '--k', alpha=0.5)

        for ind in range(self.y_data.shape[1]):
            axis.plot(self.y_data[:, ind], y_pred[:, ind],
                      marker=markers[ind], mfc='None', ls='',
                      label=self.y_labels[ind])

        axis.legend()

        axis.set_xlabel('$%s{data}$' % self.y_name)
        axis.set_ylabel('$%s{model}$' % self.y_name)

        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        axis.text(1, 1.04, 'num_components = %i' % self.num_comp,
                  transform=axis.transAxes, ha='right')

        return fig, axis

    def cross_validation(self, num_groups=10):
        perm = np.random.permutation(self.data.shape[0])
