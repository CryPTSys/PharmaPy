# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:21:24 2020

@author: dcasasor
"""

import numpy as np


def gaussian(x, mu=0, sigma=1, ampl=1):
    func = ampl/sigma/np.sqrt(2*np.pi) * np.exp(-0.5*((x - mu) / sigma)**2)

    return func


def multiple_gaussian(x, mus, sigmas, ampls, separated=False):

    num_mu = len(mus)
    num_sigma = len(sigmas)
    num_ampl = len(ampls)

    num_x = len(x)

    equal = num_mu == num_sigma == num_ampl
    if not equal:
        raise RuntimeError("'mus', 'sigmas' and 'ampls' must have the same "
                           "number of elements")

    gaussians = np.zeros((num_mu, num_x))
    for ind in range(num_mu):
        gaussians[ind] = gaussian(x, mus[ind], sigmas[ind], ampls[ind])

    gaussians = gaussians.T

    if separated:
        return gaussians
    else:
        return gaussians.sum(axis=1)


def gaussian_dparam(x, mu=0, sigma=1, ampl=1):
    gauss = gaussian(x, mu, sigma, ampl)

    df_dmu = gauss * (x - mu) / sigma**2
    df_dsigma = gauss / sigma * (((x - mu) / sigma)**2 - 1)
    df_dampl = gauss / ampl

    return df_dmu, df_dsigma, df_dampl


def gaussian_dx(x, mu=0, sigma=1, ampl=1):
    gauss = gaussian(x, mu, sigma, ampl)

    df_dx = -gauss * (x - mu) / sigma**2

    return df_dx


def gaussian_dxdx(x, mu=0, sigma=1, ampl=1):
    gauss = gaussian(x, mu, sigma, ampl)

    df_dxdx = gauss / sigma**2 * (((x - mu) / sigma)**2 - 1)

    return df_dxdx


def gauss_dparam_mult(x, mus, sigmas, ampls):
    num_mu = len(mus)
    num_sigma = len(sigmas)
    num_ampl = len(ampls)

    num_x = len(x)

    equal = num_mu == num_sigma == num_ampl
    if not equal:
        raise RuntimeError("'mus', 'sigmas' and 'ampls' must have the same "
                           "number of elements")

    dgauss_dmu = np.zeros((num_x, num_mu))
    dgauss_dsigma = np.zeros_like(dgauss_dmu)
    dgauss_dampl = np.zeros_like(dgauss_dmu)

    for ind in range(num_mu):
        df_dmu, df_dsigma, df_dampl = gaussian_dparam(x, mus[ind], sigmas[ind],
                                                      ampls[ind])

        dgauss_dmu[:, ind] = df_dmu
        dgauss_dsigma[:, ind] = df_dsigma
        dgauss_dampl[:, ind] = df_dampl

    dgauss_dparam = np.hstack((dgauss_dmu, dgauss_dsigma, dgauss_dampl))

    return dgauss_dparam


def gauss_dx_mult(x, mus, sigmas, ampls):
    num_mu = len(mus)
    num_sigma = len(sigmas)
    num_ampl = len(ampls)

    num_x = len(x)

    equal = num_mu == num_sigma == num_ampl
    if not equal:
        raise RuntimeError("'mus', 'sigmas' and 'ampls' must have the same "
                           "number of elements")

    dgauss_dx = np.zeros((num_mu, num_x))

    for ind in range(num_mu):
        dgauss_dx[ind] = gaussian_dx(x, mus[ind], sigmas[ind], ampls[ind])

    dgauss_dx = dgauss_dx.sum(axis=0)

    return dgauss_dx


def gauss_dxdx_mult(x, mus, sigmas, ampls):
    num_mu = len(mus)
    num_sigma = len(sigmas)
    num_ampl = len(ampls)

    num_x = len(x)

    equal = num_mu == num_sigma == num_ampl
    if not equal:
        raise RuntimeError("'mus', 'sigmas' and 'ampls' must have the same "
                           "number of elements")

    dgauss_dxdx = np.zeros((num_mu, num_x))

    for ind in range(num_mu):
        dgauss_dxdx[ind] = gaussian_dxdx(x, mus[ind], sigmas[ind], ampls[ind])

    dgauss_dxdx = dgauss_dxdx.sum(axis=0)

    return dgauss_dxdx
