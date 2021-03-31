
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
# import jax.numpy as jnp
# import autograd.numpy as jnp
from numpy.linalg import norm

eps = np.sqrt(np.finfo(float).eps)

def dx_jac_x(x, abstol, reltol, eps):
    sigma_zero = 1
    crit_1 = np.abs(x) * np.sqrt(eps)

    weights = 1 / (reltol * x + abstol)
    crit_2 = sigma_zero / weights

    dx = np.maximum(crit_1, crit_2)

    # print(dx)
    return dx

def dx_jac_p(p, abstol, reltol, eps):
    dx = np.abs(p) * np.sqrt(max(reltol, eps))
    return dx

def numerical_jac(func, x, args=(), dx=None, abs_tol=None, rel_tol=None):

    if dx is None:
        dx = np.ones_like(x) * eps
    elif callable(dx):
        dx = dx(x, abs_tol, rel_tol, eps)
    else:
        dx = np.ones_like(x) * dx

    f_eval = func(x, *args)

    num_x = len(x)
    num_f = len(f_eval)

    jac = np.zeros((num_f, num_x))
    delx = np.zeros_like(x)

    for i in range(num_x):
        delx[i] = dx[i]
        jac[:, i] = (func(x + delx, *args) - f_eval)/dx[i]
        delx[i] = 0

    return jac


def numerical_jac_central(func, x, rel_tol, abs_tol, dx=None, args=()):

    if dx is None:
        dx = np.ones_like(x) * eps
    elif callable(dx):
        dx = dx(x, abs_tol, rel_tol, eps)
    else:
        dx = dx * np.ones_like(x)

    num_x = len(x)
    jac = []
    delx = np.zeros_like(x)

    for j in range(num_x):
        delx[j] = dx[j]
        jac.append((func(x + delx, *args) - func(x - delx, *args)) /2. / dx[j])
        delx[j] = 0

    return np.column_stack(jac)


def numerical_jac_data(func, x, args=(), dx=None):

    if dx is None:
        dx = eps
    else:
        dx = dx

    f_eval = func(x, *args)
    num_data = len(f_eval)
    num_states = len(x)

    jac = np.zeros((num_data, num_states))
    delx = np.zeros_like(x)

    for i in range(num_states):
        delx[i] = dx
        jac[:, i] = (func(x + delx, *args) - f_eval)/dx
        delx[i] = 0

    return jac


def numerical_jacv(func, x, v, args=()):
    """ Function to evalute the right product J*v. After Hindmarsh and Serban
    (2020) (CVODEs 5.1.0 manual, sec. 2.1).

    It is not very accurate calculation (compared with performing J*v directly)
    but accurate enough for an iterative linear method such as GMRS


    Parameters
    ----------
    func : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    args : TYPE, optional
        DESCRIPTION. The default is ().

    Returns
    -------
    jac_v : TYPE
        DESCRIPTION.

    """
    sig = 1/norm(v)

    f_eval = func(x, *args)
    jac_v = (func(x + sig*v, *args) - f_eval) / sig

    return jac_v


def fun(x):
    f1 = x[0]**2 - 1/2*x[1]**3
    f2 = x[0] + jnp.sqrt(x[1])

    fx = jnp.array([f1, f2])

    return fx


def jac_fun(x):
    x1, x2 = x
    dim = len(x)
    jac = np.zeros((dim, dim))

    jac[0, 0] = 2*x1
    jac[0, 1] = -3/2*x2**2
    jac[1, 0] = 1
    jac[1, 1] = 1/2/np.sqrt(x2)

    return jac


if __name__ == '__main__':
    from autograd import jacobian, make_jvp
    from jax import jvp

    # Autograd fns
    jac_ad = jacobian(fun)
    jacv_ad = make_jvp(fun)


    # Nominal x
    x_test = np.array([1., 2.])

    # Evaluate jacs
    jacfun_eval = jac_fun(x_test)
    # jacauto_eval = jac_ad(x_test)
    jacnum_eval = numerical_jac(fun, x_test)

    # Evaluate J*v
    v_test = np.array([0.5, 0.5])
    jacv_analytic = np.dot(jacfun_eval, v_test)
    jacv_numeric = numerical_jacv(fun, x_test, v_test)
    _, jacv_autograd = jacv_ad(x_test)(v_test)
    # _, jacv_jax = jvp(fun, (x_test,), (v_test,))
