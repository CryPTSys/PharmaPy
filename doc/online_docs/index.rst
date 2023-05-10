.. PharmaPy documentation master file, created by
   sphinx-quickstart on Fri Feb 11 15:06:38 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PharmaPy's documentation!
====================================

PharmaPy is an open-source library for the analysis of pharmaceutical manufacturing systems. Some of its features are:

* Fully dynamic models (ODEs and DAEs) of commonly found unit operations on the drug substance side of pharmaceutical manufacturing  
* Start-up modeling, disturbance analysis
* Fully sequential-modular approach: each unit is simulated individually in a pre-defined sequence
* Allows decoupling of continuous/discontinuous sections of a flowsheet
* Flexible modeling for batch/hybrid/continuous flowsheets
* Utilize robust numerical integrators: SUNDIALS through python package Assimulo (ODE/DAE simulation)
* In-house implementation of the Levenberg-Marquardt algorithm for Parameter Estimation
* Simulate flowsheets, estimate kinetic parameters, optimize process conditions with external tools

How to cite us:

* Casas-Orozco D, Laky D, Wang V, Abdi M, Feng X,  Wood E,  Laird C, Reklaitis, G V, Nagy, Z K. *PharmaPy: An object-oriented tool for the development of hybrid pharmaceutical flowsheets.* Computers & Chemical Engineering. 2021;153:107408::

        @article{Casas-Orozco2020,
          author = {Casas-Orozco, Daniel and Laky, Daniel and Wang, Vivian and Abdi, Mesfin and Feng, X. and Wood, E. and Laird, Carl and Reklaitis, Gintaras V. and Nagy, Zoltan K.},
          doi = {10.1016/j.compchemeng.2021.107408},
          issn = {00981354},
          journal = {Comput. Chem. Eng.},
          month = {oct},
          pages = {107408},
          title = {{PharmaPy: An object-oriented tool for the development of hybrid pharmaceutical flowsheets}},
          url = {https://linkinghub.elsevier.com/retrieve/pii/S0098135421001861},
          volume = {153},
          year = {2021}
        }

Our team
==========

Developers
++++++++++

* `Daniel Casas-Orozco`_
* Daniel J. Laky
* Inyoung Hur
* Varun Sundarkumar
* Yash Barhate
* Jung Soo Rhim
* PharmaPy logo by Montgomery Smith

Purdue University Staff
+++++++++++++++++++++++

* Prof. Zoltan Nagy (PI)
* Prof. Gintaras V. Reklaitis (coPI)

.. _`Daniel Casas-Orozco`: https://github.com/dcasasor-purdue

Support
=======
PharmaPy has been made with the collaboration and support from the following institutions:

* The `CryPTSys Lab`_ at Purdue University
* The U.S. Food and Drug Administration (FDA)

.. _`CryPTSys Lab`: https://engineering.purdue.edu/CryPTSys/index.html


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   docstrings/index
   examples/index
   publications
   advanced_usage

..   examples/PFR_Batch_solved

..   installations.rst
..   general_features.rst
..   unit_operations.rst
..   simulation_executive.rst
..   advanced_features.rst
..   phases.rst
..   kinetics.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

