============
Installation
============

..  There are two ways to install PharmaPy. The first way is for casual users, intending to use the software and not edit or add modules to the software package. The second way is for developers or advanced users who intend to create and incorporate their own models into their work.

        Standard Installation
        =====================

        We recommend using python or anaconda virtual environments to control python packages effectively, including PharmaPy and its dependencies. PharmaPy can be installed in two different ways. For those who want to just use the software in the latest stable version, use the following command:

        .. testcode::

           pip install pharmapy

        This will download and install PharmaPy to the current python environment. To edit and run code, it is recommended to also install an IDE, or use jupyer notebooks. To install jupyter notebooks and its dependencies, run the following command:

        .. testcode::

           pip install jupyterlab

..
        Developer Installation
        ======================

For installation, we recommend the use of conda environments to control packages dependencies and PharmaPy. A lighweight version of conda (`miniconda`_) is probably a good option for new users of the management system.

For PharmaPy installation, you must use the source code, which is available in our `Github repository`_. Once the source code is downloaded to the desired location, navigate (:code:`cd`) to the directory which contains the setup.py file. Then, follow the instructions on the :code:`installation_guide.txt`, to setup fresh conda environment and install PharmaPy and its dependencies.

..
        make sure your conda environment is appropriately installed and activated, then input the following commands for PharmaPy installation:
        1. conda install --file requirements.txt -c conda-forge
        2. python setup.py develop

.. _Github repository: https://github.com/CryPTSys/PharmaPy/tree/develop
.. _miniconda: https://github.com/CryPTSys/PharmaPy/

Once the software is installed, install and/or use your preferred IDE or text editor to construct PharmaPy flowsheets. For instance, on an active conda environment, install the `Spyder IDE`_ by doing :code:`conda -c conda-forge install spyder`, which provides a nice development environment very well suited for scientific computing. 

.. _Spyder IDE: https://github.com/spyder-ide/spyder

Tutorials in the format of Jupyter notebooks are available for users and developers getting started with PharmaPy. Also, on this site, documentation for all unit operations is available.
