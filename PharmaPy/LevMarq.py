#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 20:23:12 2019

@author: dcasasor
"""

from numpy import ones_like, inner, diag, asarray, maximum
from numpy.linalg import solve, norm, inv


lm_header = [
    '{:<40}'.format('-'*60),
    "{:<7} {:<10} {:<10} {:<10} {:<10}".format('eval', 'fun_val', '||step||', 'gradient', 'dampening_factor'),
    '{:<40}'.format('-'*60)]


def levenberg_marquardt(x, func, deriv, fletcher_modif=False, max_fun_eval=100,
                        eps_1=1e-8, eps_2=1e-8, tol_fun=1e-12,
                        full_output=False, mu=None, d_diag=None,
                        args=(), verbose=False):
    """
    Optimize function using the Levenberg-Marquardt algorithm

    Parameters
    ----------
    x : array-like
        1D array containing the decision variables.
    func : callable
        function with signature func(x, *args), which returns a n_data-sized
        vector of residuals (x_data - x_model).
    deriv : callable
        function that calculates the derivative of func with respect to x.
        It has the same signature as func, and returns a n_data x n_x matrix
        containing the jacobian of func
    fletcher_modif : bool, optional
        DESCRIPTION. The default is False.
    max_fun_eval : int, optional
        maximum number of function evaluations. The default is 100.
    eps_1 : float, optional
        stoping criteria for the gradient. The default is 1e-8.
    eps_2 : float, optional
        stopping criterion for step size. The default is 1e-8.
    tol_fun : float, optional
        stoping criterion for the objective function . The default is 1e-12.
    full_output : bool, optional
        If true, the returned result includes optimum x, covariance of x
        and a dictionary with details about the optimization.
        The default is False.
    lambd_zero : float, optional
        multiplier for the first value of mu. The default is 1e-2.
    args : tuple, optional
        additional arguments to pass to func and deriv callables.
        The default is ().
    verbose : bool, optional
        If True, a table with the current value of the objective, gradient
        and step size is shown as the optimization proceeds.
        The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    nu = 2
    beta = nu
    x = asarray(x)

    num_iter = 0

    fun = func(x, *args)
    jac = deriv(x, *args)

    a_matrix = inner(jac, jac)  # Hessian approximation
    b_vector = inner(jac, fun)  # gradient

    if d_diag is None:
        if fletcher_modif:
            d_scaling = norm(jac, axis=1)
            d_scaling = max(d_scaling) * ones_like(x)
        else:
            d_scaling = ones_like(x)
    else:
        d_scaling = d_diag

    if mu is None:
        mu = 1e-2 * max(diag(a_matrix))  # after Nielsen (1999)

    num_feval = 0

    if verbose:
        print('\n'.join(lm_header))
        print("{:<7} {:<10.3e} {:<10} {:<10.3e} {:<10.3e}".format(
            num_feval, norm(fun)**2, '---', norm(b_vector), mu))

    while num_feval < max_fun_eval:
        d_diag = diag(d_scaling)
        lm_step = solve(a_matrix + mu * inner(d_diag.T, d_diag), -b_vector)

        if norm(lm_step) < eps_2 * norm(x):
            reason = 'Small step'
            break

        x_new = x + lm_step
        fun_new = func(x_new, *args)
        jac_new = deriv(x_new, *args)

        num_feval += 1

        sq_old = norm(fun)**2
        sq_new = norm(fun_new)**2
        rho = (sq_old - sq_new) / (inner(lm_step, mu * lm_step - b_vector))

        if rho > 0:  # update x if function is decreased
            x = x_new
            fun = fun_new
            jac = jac_new

            a_matrix = inner(jac, jac)  # Hessian approximation
            b_vector = inner(jac, fun)

            if fletcher_modif:
                d_scaling = maximum(norm(jac, axis=1), d_scaling)

            num_iter += 1

            if norm(b_vector) < eps_1:
                reason = 'Small gradient'
                break

            mu = mu * max(1/3, 1 - (beta - 1) * (2*rho - 1)**3)  # After Nielsen
            nu = 2
        else:
            mu = mu * nu  # decrease step size
            nu = 2 * nu

        beta = nu

        if verbose:
            if num_feval % 50 == 0:
                print('\n'.join(lm_header))

            print("{:<7} {:<10.3e} {:<10.3e} {:<10.3e} {:<10.3e}".format(
                num_feval, sq_new, norm(lm_step), norm(b_vector), mu))

        if norm(fun_new) < tol_fun:
            break
            reason = 'Value of the objective function lower than tolerance'

    else:
        reason = 'Maximum iterations exceeded'

    covar_x = inv(a_matrix)

    if verbose:
        print('{:<40}'.format('-'*60))
        print()

    if full_output:
        lm_par = {'d_diag': d_scaling, 'mu': mu}
        if max_fun_eval == 0:
            output_dict = {'x': x, 'fun': fun, 'jac': jac,
                           'num_iter': num_iter, 'num_fun_eval': num_feval,
                           'lm_params': lm_par}
        else:
            output_dict = {'x': x, 'fun': fun, 'jac': jac,
                           'norm_step': norm(lm_step),
                           'stop_criterion': reason, 'num_iter': num_iter,
                           'num_fun_eval': num_feval, 'lm_params': lm_par}

        return x, covar_x, output_dict
    else:
        return x
