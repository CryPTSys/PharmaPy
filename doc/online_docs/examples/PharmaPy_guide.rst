===========================
PharmaPy Step-by-step Guide
===========================

Introduction
============
This section has been written for the purpose of further assisting the user experience with PharmaPy code. Thus, this section should be used as an instructional manual that is to be used in conjunction with the "Process optimization with PharmaPy" notebook.

Preprocess
==========
In this section, decisions that are not specific to a single unit operation must be made. The most important of theses decisions is specifying what chemical species will be used as well as their physical properties.

Chemical selection
------------------

1.	Specify the chemical properties of the involved compounds in a JSON file. Be sure to define the file path for the program to access the properties. Each entry for the species should be in the following template:
   
.. code-block:: json

	{
	“Species_Name”:
		{
		
		“mw”: molecular weight of the species in units of [g/mol],
		
		“t_crit”: critical temperature of the species in units of [K],
		
		“p_crit”: critical pressure of the species in units of [Pa],

		“cp_liq”: [coefficients for the polynomial form of the liquid specific heat of the species in units of [J/K/mol]], # cP(T) = A + BT + CT**2 + DT**3 + ET**4
		
		“cp_solid”: [coefficients for the polynomial form of the solid specific heat of the species in units of [J/K/mol]],

		“rho_liq”: density of the species in liquid form in units of [kg/m^3],

		“rho_solid”: density of the species in solid form in units of [kg/m^3],

		“visc_liq”: [coefficients for the fourth power exponential form for viscosity of the species in units of [kg/m/s]], # mu(T) = A * exp(B/T + CT + DT**2

		“p_vap”: [coefficients for the Antoine equation for vapor pressure of the species in units of [Pa]], # log P = A - B/(C - T)

		“mol_vol”: molar volume of the species in units of [m^3/mol],

		“delta_hvap”: enthalpy of vaporization of the species in units of [J/mol],

		“tref_hvap”: enthalpy of vaporization of the species from a reference temperature in units of [J/mol],

		“surf_tension”: surface tension value of the species in units of [N/m],

		“source”: source for the chemical properties listed above
		}
	}

It should be noted that supplying the values for the molecular weight, liquid density, and solid density of the species is mandatory for the analysis to be run. However, it should also be noted that while the basis analysis with PharmaPy may run without the other values, other in-depth analysis of the system may be inaccurate without the other chemical and thermodynamic properties.

The other properties, :code:`t_crit`, :code:`cp_liq`, :code:`cp_solid`, :code:`p_vap`, :code:`delta_hvap`, and :code:`tref_hvap`, are needed for the Drying unit operations. However, they are not needed for the scope of this example.

2.	Define the order of operations in string format. The flow of the unit operations should be specified using the :code:`-->` string.

.. testcode::

	('R01 --> HOLD01 --> CR01 --> F01')

In the above example, the system process is defined as starting from a reactor (:code:`R01`) to a holding tank (:code:`HOLD01`), then going to a crystallizer (:code:`CR01`) and finally going to a filtration unit (:code:`F01`).

3.	Define a variable denoting the simulation object and using the SimulationExec’ function.

.. testcode::

	sim = SimulationExec(path_phys, flowsheet=graph).

The variable ‘path_phys’ is the string for the file path for the chemical properties (i.e. 'C:\user\Documents\..).

4. Define the chemical reactions that occur with the species in the process. The reactions that can be specified are noted with strings.

.. testcode::

	rxns = [‘A + B --> C’, ‘A + C --> D’]

In the example above, it is noted that chemical A reacts with chemical B to create chemical C. Additionally, chemical A also reacts with chemical C to create chemical D.

5. Input the chemical parameters for the products of the system.

	* Input the pre-exponential factor value(s) for the temperature dependent terms for the products.
	
		.. testcode::
		
			k_vals = np.array([2.654e4, 5.3e2])
	
	   In the above example, writing k-values using numpy array function guarantees the parameters are in the correct format. Additionally, it should be noted that the k-values dictates the nucleation rates for the products in accordance with the formulation of
	
	.. math::
	
		r_{i} = \mathbf{k}_{\mathbf{i}}\exp\left( - \frac{Ea_{i}}{RT} \right)

	* Input the activation energy value(s) for the temperature dependent terms for the products.
	
		.. testcode::
		
			ea_vals = np.array([4.0e4, 3.0e4])

	   In the above example, writing ea-values using numpy array function guarantees the parameters are in the correct format. Additionally, it should be noted that the :code:`ea-vals` values dicates the nucleation rates for the products in accordance with the formulation of:
	
	.. math::
	
		r_{i} = k_{i}\exp\left( - \frac{\mathbf{E}\mathbf{a}_{\mathbf{i}}}{RT} \right)

6. Finally, input the specified parameters into the :code:`RxnKinetics()` function.

.. testcode::

	kinetics = RxnKinetics(path=path_phys, rxn_list=rxns,k_params=k_vals, ea_params=ea_vals)

It should be noted that the input for the file path, the :code:`k_params` and the :code:`ea_params` are mandatory for the execution of the :code:`RxnKinetics()` function. However, one must also supply *either* the reaction list *or* a stochiometric matrix for the function to be created successfully.

Reactor
=======
In this section, decisions that directly affect the reactor portion of the system process are defined.

Reactor Setup
-------------

1. The main input for the reactor is the flow rate. While there may be several different methods of establishing the volume flow rate of a reactor, it can generally be calculated as :math:`\frac{vol_{liq}}{tau_{R01}}`.

2. Determine the percentage composition for the initial solution in the reactor.
	
	.. testcode::
	
		w_init = np.array([0,0,0,0,1])

The example shows that 100% of the composition in the reactor is the solvent, which is represented in the last column.

3. Set the initial temperature for the reactor in units of [K].

4. With the previous parameters, use the :code:`LiquidPhase()` function to define the physical reactions of the state in the reactor(s).

.. testcode::

	liquid_init = LiquidPhase(path_phys, init_temp, mass_frac=w_init,vol=vol_liq)

In the above example, the first input denotes the file path for the chemical properties, the second input denotes the starting temperature, the third input denotes the starting composition, and the final input denotes the volume.

Cooling
-------

1. Set the temperature you want the reactor to be set to in units of [K].


2. Use the :code:`CoolingWater()` function to create a cooling water object

.. testcode::
	
	cw = CoolingWater(mass_flow, temp_in=temp_set_R01)

In the above example, the first input denotes the rate at which the cooling water will be circulated, in units of [kg/s]. The second input is the temperature at which the incoming cooling water will be set to. Note that the cooling water supplied to other unit operations, such as crystallizers, are instantiated in the same manner.

Main Reactor
------------

1. Set what chemical components will be introduced into the reactor. The input amount can be set either as mass fractions or molar concentrations.

.. testcode::
	
	c_in = np.arrary([0.33, 0.33, 0, 0, 0])
	
In the above example, the initial concentration is defined as an array. For this example, the chemical species in the "compound_database" JSON file are ordered as species A, B, C, D, and the solvent. Thus, in the example, the in example, the initial concentration has a molar concentration of 0.33 for species A and 0.33 for species B, and none for the other chemicals components.

2. Set the temperature at which the chemical components will be introduced at.

3. Use the previous parameters with the :code:`LiquidStream()` function to create the input liquid object.

.. testcode::

	LiquidStream(path_phys, temp_in, mole_conc=c_in, vol_flow=vol_flow, name_solv='solvent')

In the above example, the first input denotes the file path for the chemical properties. The second input denotes the temperature at which the chemical components are introduced. The third input denotes the molar concentrations of the introduced chemicals. The fourth input denotes the rate of flow into the reactor. The final input denotes the string name of the solvent in the JSON file of chemical properties.

4. Set the diameter of opening through which the chemicals are
   introduced to the reactor.

5. Assign the reactor type to the SimulationExec object made in the first step.

.. testcode::

	sim.R01 = PlugFlowReactor(diam_in, num_discr=50, isothermal=False)

In this example, a plug flow reactor is being used. Thus, the :code:`PlugFlowReactor()` function is used. The first input denotes the diameter of the opening through which the chemicals are introduced. The second input denotes the number of finite volumes used to discretize the volume coordinates. Usually set to 50. The final input is a boolean value to determine whether or not the reactor is isothermal.

6. Assign the CoolingWater object to the SimulationExec object’s R01.Utility value.

.. testcode::

	sim.R01.Utility = cw

7. Assign the RxnKinetics object to the SimulationExec object’s R01.Kinetics value.

.. testcode::

	sim.R01.Kinetics = kinetics

8. Assign the LiquidStream object to the SimulationExec object’s R01.Inlet value.

.. testcode::

	sim.R01.Inlet = liquid_in

9. Assign what phases of matter are present in the reactor to the SimulationExec object’s R01.Phases value. For the most part, this will be the LiquidPhase object.

.. testcode::

	sim.R01.Phases = liquid_init

Holding Tank
============

For this example, the holding tank is defined with a single use of the :code:`DynamicCollector()` function, assigned to the SimulationExec object’s H01 value. There are no parameter inputs. This section will be updated as needed.

.. testcode::
	
	sim.HOLD01 = DynamicCollector()

Crystallizer
============

1. Define the parameters for nucleation (primary and secondary), growth, and dissolution in the crystallizer. These values will be input to the :code:`CrystKinetics()` function at a later step. Furthermore, the nucleation, growth, and dissolution kinetics are calculated using the following equations:

	* Primary nucleation, growth, and dissolution
	
		.. math::
		
			f = A\exp\left( - \frac{B}{RT} \right){\sigma|\sigma|}^{(C - 1)}

	* Secondary nucleation
	
		.. math::
	
			f = A\exp\left( - \frac{B}{RT} \right){\sigma|\sigma|}^{(C - 1)}\left( k_{v}\mu_{3} \right)^{D}

2. Define the solubility constants of the process. Use the array function from numpy to express the constants as an array. The given constants are used to calculate the temperature dependent solubility in the form of the following equation:

.. math::

	S(T) = A + BT + CT^{2} + DT^{3}
	
The solubility is calculated in units of [kg/m^3]

3. Setup up the distribution of the API in the crystallizer using numpy.geomspace, which sets up a logarithmically spaced list of numbers between a given start and stop.


.. testcode::

	x_gr = np.geomspace(1, 1500, num=35)

The above code gives a list of 35 entries where the values are logarithmically spaced from 1 to 1500.

4. Setup the initial distribution of API. If not seeded, it should all be 0 values. Must have the same dimension as :code:`x_gr`.

5. Input the previous values into to the :code:`SolidPhase()` function to define the crystallization kinetics for the target API.

.. testcode::

	solid_cry = SolidPhase(path_phys, x_distrib=x_gr,distrib=distrib_init,mass_frac=[0,0,1,0,0])

In the above example, the first input denotes the file path for the JSON file with all chemical properties. The second input denotes the distribution of the API. The third input denotes the initial distribution. Finally, the fourth input denotes the mass fraction of what species will be in solid phase. For this example, only the third species, the chemical C, was to be solidified in the crystallizer. Thus, the mass fraction list only has a non-zero value in the third column.

6. Create an array for the temperature profile. Each element of the array should be a two-element list, denoting the start and end temperature of each section of the temperature profile.

7. Select the runtime for the crystallizer. Typically set at twice the length of the reactor residence time.

8. Input the previous two variables into the :code:`PiecewiseLagrange()` function to put the temperature profile in the format that is necessary for the function for the crystallizer.

.. testcode::

	lagrange_fn = PiecewiseLagrange(runtime_cryst, temp_program)

9. Assign the function for the crystallizer to the SimulationExec object’s CR01 value. In this example, a batch crystallizer is being used. Thus the :code:`BatchCryst()` function is used.

.. testcode::

	sim.CR01 = BatchCryst(target_comp='C', method='1D-FVM', scale=1e-9,controls={'temp': lagrange_fn.evaluate_poly})

In the above example, the first input denotes the string name of the chemical that we are tracking in the crystallizer. In this example, we are tracking the vaguely named chemical C. The second input denotes the method used to solve the system. In this example, we are using the 1D Finite Volume Element method. The third input denotes the scale with which everything is calculated. Thus, it applies a 1e-9 multiplier to the inputs. The fourth input denotes what temperature profile, or in other cases, antisolvent addition method for the crystallizer.

10. Assign the CrystKinetics function to the SimulationExec object’s CR01.Kinetics value.

.. testcode::

	sim.CR01.Kinetics = CrystKinetics(solub_cts, nucl_prim=prim,nucl_sec=sec, growth=growth, dissolution=dissol)

In the above example, the first input denotes the solubility constants of the process. The second input denotes the primary nucleation parameters. The third input denotes the secondary nucleation parameters. The fourth input denotes the growth parameters. The fifth input denotes the dissolution parameters.

11. Assign the CoolingWater function to the SimulationExec object’s CR01.Utility value.

.. testcode::

	sim. CR01.Utility = CoolingWater(mass_flow=1, temp_in=283.15)

In the above example, the first input denotes the rate of flow for the temperature regulation of the crystallizer. The second input denotes the temperature at which the flowing water is at.

12. Assign the SolidPhase object to the SimulationExec object’s CR01.Phases.

.. testcode::

	sim.CR01.Phases = solid_cry

Filter
======

1. Determine the value for :math:`\alpha`, the cake resistivity, for the process.

2. Determine the diameter of the filter in the process.

3. Determine the resistance of the media in the filter.

4. Input the previous values into the SimulationExec’s F01 value using the :code:`Filter()` function.

.. testcode::

	sim.F01 = Filter(diam, alpha, Rm)

In the above example, the first input denotes the diameter of the filter. The second input denotes the cake resistivity of the API. The third input denotes the mesh resistance of the filter.

Solving the Flowsheet
=====================

1. Define the keyword arguments (kwargs) for solving the flowsheet. For this example, the main kwargs are for the runtime.

.. testcode::

	runargs_R01 = {'runtime': runtime_reactor}
	
	sundials = {'maxh': 60}

	runargs_hold = {'runtime': runtime_reactor}

	runargs_CR01 = {'runtime': runtime_cryst, 'sundials_opts': sundials}

	runargs_F01 = {'runtime': None}
	
	run_kwargs = {'R01': runargs_R01, 'HOLD01': runargs_hold, 'CR01':
	
	runargs_CR01, 'F01': runargs_F01}

In the above example, the runtime for the Reactor, Holding Tank, Crystallizer, and the Filter are implemented. It should also be noted that for the Crystallizer, an additional kwarg with ‘sundials’ is implemented. This is added to ensure a smooth graphing for the steep curves which are characteristic in a crystallizer.

2. Input the kwargs into the :code:`SolveFlowsheet()` attribute of the SimExec's object.

.. testcode::

	sim.SolveFlowsheet(kwargs_run=run_kwargs)

3. The results of the flowsheet can then be plotted using the :code:`plot_profiles` attribute of the unit operations in the SimExec's object.

.. testcode::

	sim.CR01.plot_profiles()

In the example above, the command will plot the moments :math:`\mu_{i}` where :math:`i = 0,1,2,3`, the temperature profile, and the concentration/solubility/supersaturation curves.
