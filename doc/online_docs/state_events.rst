State events
============

State events are passed as Python dictionaries. Simple state events based directly on model states (i.e. :math:`f(y) = y`) can be passed as shown for the following structure. For detecting a given temperature, one would do:

.. testcode::

   my_state_event = {'state_name': 'temperature', 'value': 400} 

If a given state is index, e.g. concentration for a given component, the dictionary also needs to have the `state_idx` field. For example, for detecting the concentration the firs component for a reactor model, the dictinary to be passed would be:

.. testcode::
   my_state_event = {'state_name': 'mole_conc', 'state_idx': 0, 'value': 0.2} 


