====================
Advanced features
====================

State events
============

State events are passed as Python dictionaries. Simple state events based directly on model states (i.e. :math:`f(y) = y`) can be passed as shown for the following structure. For detecting a given temperature, one would do:

.. testcode::

   my_state_event = {'state_name': 'temperature', 'value': 400} 

If a given state is not scalar but has to be indexed, e.g. concentration for a given component, the dictionary also needs to have the :code:`state_idx` field. For example, if the simulation is to be stopped when the  concentration the first component for a reactor model reaches a reference value, the dictionary describing the state event would be:

.. testcode::

   my_state_event = {'state_name': 'mole_conc', 'state_idx': 0, 'value': 0.2} 

More advanced usage of state events is allowed by passing a callable directly to PharmaPy. This callable will be able to make full use of all the instantaneous state and derivative information, which is passed by PharmaPy at each integration step. In this case, the function is passed using a dictionary with the keyword :code:`callable`:

.. testcode::

   my_state_event = {'callable': my_function}

where the passed callable function must have the signature :code:`my_function(time, states, sdot, **kwargs)` and must return a scalar whose sign changes only when the event is detected. The passed :code:`state` and :code:`sdot` arguments will be dictionaries that have the names of the states as keys. Any keyword arguments can be optionally specified in the state event dictionary, e.g.:

.. testcode::

   my_state_event = {'callable': my_function, 'kwargs': {...}}

An example of a callable passed as a state event is when solubility wants to be monitored. A callable could have the following form:

.. testcode::

   def my_callable(time, y, ydot, a, b, c):
       solubility = a + b * y['temp'] + c * y['temp']**2  # a, b, and c are solubility constants
       event = solubility - y['mass_conc'][1]  # Let's say it is a binary system where the first component is the solvent and the second one is the API
       
       return event

In this case, the returned :code:`event` variable will be positive until the solubility limit is reached. When that happens, its sign change will be detected by PharmaPy and the integration will be interruped.

Interpolators
===============

Input trajectories can be specified via interpolators. To this purpose, PharmaPy contains a Lagrange polynomial represention that describes a time-dependent input :math:`u(t)` as:

.. math::

   u(t) = \sum_{i = 1}^{ord} u_{i, k} \ell_i^{(ord)} (\tau^{(k)}), \quad t \in [t_{k - 1}, t_k], \ k \in \{1, \ldots, n_{interv}\},

for a set of user-provided points :math:`u_{i, k}, \ i \in \{1, \ldots, ord\}` within the interval :math:`k`.  :math:`\tau` is a normalized time variable given by:

.. math::
   \tau = \frac{t - t_{k - 1}}{t_{k} - t_{k - 1}}, \quad t \in [t_{k - 1}, t_k]

and :math:`\ell^{(ord)}` are Lagrange interpolation polynomials given by:

.. math::
   \ell_{i}^{(ord)} =
   \begin{cases}
       1, & ord = 1, \\
       \prod_{j = 1, j \neq i}^{ord} \frac{\tau - \tau_j}{\tau_i - \tau_j}  & ord \geq 2,
   \end{cases}

where :math:`ord` represents the order of the interpolation (1 for piecewise constant, 2 for piecewise linear, etc).

For example, if a given flowrate wants to be specified as a piecewise constant function, a PharmaPy interpolator can be specified as: 

.. testcode::

   from PharmaPy.Interpolation import PiecewiseLagrange

   time_hor = 3600  # total time [s]
   flowrates = [0.1, 0.2, 0.1, 0.6]  # kg/s
   interpolator = PiecewiseLagrange(time_hor, flowrates, order=1)

In this particular case, the horizon time will be split into equally sized, 15-min (900 s) bins with their corresponding four specified flows as specified in the :code:`flowrates` variable. User-defined time marks can also be passed as a list or NumPy array by using the :code:`time_k` argument of the :code:`PiecewiseLagrange` interpolator, which needs to be of size :code:`n_y + 1`, where :code:`n_y` is the vector of interpolated values (:code:`flowrates` in this example).

Piecewise linear interpolators can also be used. In this case, the passed known values must be arranged into a numpy 2-D array, and the interpolation order will be 2. For example, a linear piecewise temperature profile would be constructed as:

.. testcode::

   from PharmaPy.Interpolation import PiecewiseLagrange

   time_hor = 3600  # total time [s]
   temperatures = np.array([[360, 345],
                            [345, 330],
                            [330, 318],
                            [318, 295]])  # K
   interpolator = PiecewiseLagrange(time_hor, temperatures, order=2)

Note that the values on the second column always match the value of the first column in the next raw, for continuity purposes. Higher orders will follow the same structure, where each row will represent a subinterval and the number of columns will dictate the interpolation order, which must be passed using the :code:`order` argument.

