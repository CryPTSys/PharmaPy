.. PharmaPy documentation master file, created by
   sphinx-quickstart on Fri Feb 11 15:06:38 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PharmaPy's documentation!
====================================

.. image:: logo.jpeg

PharmaPy is an open-source library for the analysis of pharmaceutical manufacturing systems. WE WILL NEED TO BETTER DESCRIBE THE PLATFORM 

Some of its features are:

* Fully dynamic modeling (ODEs and DAEs)
* Start-up modeling, disturbance analysis
* Fully sequential-modular approach: each unit is simulated individually in a pre-defined sequence
* Allows decoupling of continuous/discontinuous sections of a flowsheet
* Flexible modeling for batch/hybrid/continuous flowsheets
* Utilize robust numerical integrators: SUNDIALS through python package Assimulo (ODE/DAE simulation)
* In-house implementation of the Levenberg-Marquardt algorithm for Parameter Estimation
* Simulate flowsheets, estimate kinetic parameters, optimize process conditions with external tools

How to cite us

* Casas-Orozco D, Laky D, Wang V, et al. PharmaPy: An object-oriented tool for the development of hybrid pharmaceutical flowsheets. Comput Chem Eng. 2021;153:107408:

  .. testcode::
        
        @article{Casas-Orozco2020,
          author = {Casas-Orozco, Daniel and Laky, Daniel and Wang, Vivian and Abdi, Mesfin and Feng, X. and Wood, E. and Laird, Carl and Reklaitis, Gintaras V. and Nagy, Zoltan K.},
          doi = {10.1016/j.compchemeng.2021.107408},
          file = {:C\:/Users/dcasasor/OneDrive - purdue.edu/postdoc/academic/bib/our_publications/2021-Casas_et_al_PharmaPy.pdf:pdf},
          issn = {00981354},
          journal = {Comput. Chem. Eng.},
          month = {oct},
          pages = {107408},
          title = {{PharmaPy: An object-oriented tool for the development of hybrid pharmaceutical flowsheets}},
          url = {https://linkinghub.elsevier.com/retrieve/pii/S0098135421001861},
          volume = {153},
          year = {2021}
        }


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation

   :caption Class-level documentation:

   unit_operations.rst

   :caption Advanced use:

   state_events

..   general_features.rst
..   unit_operations.rst
..   simulation_executive.rst
..   advanced_features.rst




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
