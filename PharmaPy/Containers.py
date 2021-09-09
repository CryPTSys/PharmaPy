# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:26:50 2020

@author: dcasasor
"""

from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

from PharmaPy.Phases import LiquidPhase, SolidPhase, classify_phases
from PharmaPy.Streams import LiquidStream, SolidStream
from PharmaPy.MixedPhases import Slurry, SlurryStream

from PharmaPy.NameAnalysis import get_dict_states
from PharmaPy.Crystallizers import SemibatchCryst
from PharmaPy.NameAnalysis import get_dict_states

from scipy.optimize import newton, fsolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import copy


class Mixer:
    def __init__(self, phases=(), vol=None, temp_refer=298.15):

        self._Inlets = []

        self.oper_mode = None

        self.elapsed_time = 0
        self.vol = vol

        self.temp_refer = temp_refer

        self.nomenclature()
        self.is_continuous = None
        self.timeProf = None

        self.type_out = None

    @property
    def Inlets(self):
        return self._Inlets

    @Inlets.setter
    def Inlets(self, inlets):
        if isinstance(inlets, (list, tuple)):
            self._Inlets += list(inlets)
        else:
            self._Inlets.append(inlets)

        # if update_names:
        self.names_upstream.append(None)
        self.bipartite.append(None)

    def nomenclature(self):
        self.names_states_in = ['mass_frac', 'mass', 'mass_flow', 'temp']
        self.names_states_out = self.names_states_in
        self.names_upstream = []
        self.bipartite = []

    def get_inputs(self, inlets=None):
        if inlets is None:
            inlets = self.Inlets

        num_species = inlets[0].num_species

        timeseries_flag = [inlet.y_upstream is not None
                           for inlet in inlets]

        flow_flags = [hasattr(inlet, 'mass_flow') for inlet in inlets]

        is_mass = not(any(flow_flags))
        is_flow = all(flow_flags)

        if not is_mass and not is_flow:
            raise RuntimeError('Both mass and mass flow specified for one or '
                               'more streams. Please provide consistent '
                               'material amount specification')

        if is_mass:
            self.oper_mode = 'Batch'
        else:
            self.oper_mode = 'Continuous'

        massfracs = []
        masses = []
        temps = []

        if is_flow:
            names_in = [name for name in self.names_states_in
                        if name != 'mass']

            self.is_continuous = True

            timegrid_ref = None
            for ind, inlet in enumerate(self.Inlets):
                if inlet.y_upstream is None:
                    massfracs.append(inlet.mass_frac)
                    temps.append(inlet.temp)
                    masses.append(inlet.mass_flow)

                else:
                    if timegrid_ref is None:
                        timegrid_ref = inlet.time_upstream
                        y_inlet = inlet.y_inlet

                        self.timeProf = timegrid_ref
                    else:
                        y_inlet = inlet.InterpolateInputs(timegrid_ref)

                    input_dict = get_dict_states(self.names_upstream[ind],
                                                 num_species, 0, y_inlet)

                    di = {}
                    for key in names_in:
                        di[key] = input_dict.get(self.bipartite[ind][key])

                    massfracs.append(di['mass_frac'])
                    masses.append(di['mass_flow'])
                    temps.append(di['temp'])

            dict_inputs = {}
            dict_inputs['mass_frac'] = massfracs
            dict_inputs['mass'] = masses
            dict_inputs['temp'] = temps

            names_out = names_in

        else:
            for inlet in inlets:
                massfracs.append(inlet.mass_frac)
                temps.append(inlet.temp)

                # mass = getattr(inlet, 'mass', getattr(inlet, 'mass_flow'))
                masses.append(inlet.mass)

            masses = np.array(masses)
            massfracs = np.array(massfracs)
            dict_inputs = {'mass': masses, 'mass_frac': massfracs,
                           'temp': temps}

            # if is_mass:
            names_out = [name for name in self.names_states_out
                         if name != 'mass_flow']

            self.is_continuous = False
            self.timeProf = [0]

            # else:
            #     names_out = [name for name in self.names_states_out
            #                  if name != 'mass']

            #     self.is_continuous = True

        self.names_states_out = names_out

        return dict_inputs

    def get_inputs_solids(self):

        timeseries_flag = []

        for inlet in self.Inlets:
            if inlet.y_upstream is None:
                timeseries_flag.append(False)
            else:
                timeseries_flag.append(len(inlet.y_upstream) > 1)

        solids_flag = [hasattr(inlet, 'Solid_1') for inlet in self.Inlets]

        mass_solid = []
        mass_liquid = []

        massfrac_liq = []
        distrib_sol = []

        temps = []

        ind_solid = np.argmax(solids_flag)
        num_dist = self.Inlets[ind_solid].Solid_1.distrib.shape[0]

        if any(timeseries_flag):
            pass
        else:
            self.oper_mode = 'Batch'  # TODO: not necessarily
            for inlet in self.Inlets:
                if hasattr(inlet, 'Solid_1'):
                    mass_solid.append(inlet.Solid_1.mass)
                    mass_liquid.append(inlet.Liquid_1.mass)

                    massfrac_liq.append(inlet.Liquid_1.mass_frac)

                    if 'Slurry' in inlet.__class__.__name__:
                        distrib_sol.append(inlet.Solid_1.distrib * inlet.vol)
                    else:  # Cake
                        distrib_sol.append(inlet.Solid_1.distrib)

                    temps.append(inlet.Liquid_1.temp)
                else:
                    mass_solid.append(0)
                    mass_liquid.append(inlet.mass)

                    massfrac_liq.append(inlet.mass_frac)
                    distrib_sol.append(np.zeros(num_dist))

                    temps.append(inlet.temp)

            massfrac_liq = np.array(massfrac_liq)
            mass_solid = np.array(mass_solid)
            mass_liquid = np.array(mass_liquid)
            temps = np.array(temps)
            distrib_sol = np.array(distrib_sol)

        dict_out = {'temp': temps, 'mass_frac': massfrac_liq,
                    'mass_liq': mass_liquid, 'mass_solid': mass_solid,
                    'num_distrib': distrib_sol}

        return dict_out, ind_solid

    def energy_balance(self, u_inputs):

        massfrac_in = u_inputs['mass_frac']
        temp_in = u_inputs['temp']
        mass_in = u_inputs['mass_liq']

        h_in = []
        for ind, inlet in enumerate(self.Inlets):
            if hasattr(inlet, 'Solid_1'):
                distrib_in = u_inputs['num_distrib']
                h_in.append(inlet.getEnthalpy(temp_in[ind],
                                              mass_frac=massfrac_in[ind],
                                              distrib=distrib_in[ind]))
            else:
                h_in.append(inlet.getEnthalpy(temp_in[ind],
                                              mass_frac=massfrac_in[ind]))

        def temp_root(temp):
            h_out = self.Outlet.getEnthalpy(temp)

            balance = mass_in.dot(h_in) - sum(mass_in) * h_out

            return balance

        temp_seed = np.mean(temp_in)
        temp_bce = newton(temp_root, temp_seed)

        return temp_bce

    def balances(self, u_inputs):
        massfrac_in = u_inputs['mass_frac']
        mass_in = u_inputs['mass']
        temp_in = u_inputs['temp']

        # ---------- Material balances
        total_mass = mass_in.sum()
        massfrac = np.dot(mass_in, massfrac_in)/total_mass

        # ---------- Energy balance
        h_in = []
        for temp, mass_frac in zip(temp_in, massfrac_in):
            h_in.append(self.Liquid_1.getEnthalpy(temp, mass_frac=mass_frac))

        def temp_root(temp):
            h_out = self.Liquid_1.getEnthalpy(temp, temp_ref=self.temp_refer,
                                              mass_frac=massfrac)

            balance = mass_in.dot(h_in) - total_mass * h_out

            return balance

        temp_seed = np.mean(temp_in)
        temp_bce = newton(temp_root, temp_seed)

        return total_mass, massfrac, temp_bce

    def dynamic_balances(self, u_inputs):
        massfrac_in = u_inputs['mass_frac']
        mass_in = u_inputs['mass']
        temp_in = u_inputs['temp']

        # ---------- Material balances
        total_mass = sum(mass_in)
        masscomp_in = [(frac.T * mass).T for (frac, mass)
                       in zip(massfrac_in, mass_in)]
        massfrac = sum(masscomp_in) / total_mass[..., np.newaxis]

        # ---------- Energy balance
        h_in = []
        for temp, mass_frac in zip(temp_in, massfrac_in):
            h_in.append(self.Liquid_1.getEnthalpy(temp, mass_frac=mass_frac))

        energy_in = sum([mass * enth for (mass, enth) in zip(mass_in, h_in)])

        def temp_root(temp):
            h_out = self.Liquid_1.getEnthalpy(temp, temp_ref=self.temp_refer,
                                              mass_frac=massfrac)

            balance = energy_in - total_mass * h_out

            return balance

        temp_seed = sum(temp_in) / 2
        temp_bce = fsolve(temp_root, temp_seed)

        return total_mass, massfrac, temp_bce

    def balances_solids(self, u_inputs, ind_solids):
        mass_liquid = u_inputs['mass_liq']
        mass_solid = u_inputs['mass_solid']
        massfrac_liq = u_inputs['mass_frac']

        distrib_in = u_inputs['num_distrib']

        # temp = u_inputs['temp']

        # Material balances
        total_solid = sum(mass_solid)
        total_liquid = sum(mass_liquid)

        massfrac = np.dot(mass_liquid, massfrac_liq) / total_liquid

        # Physical properties
        phase_wsolids = self.Inlets[ind_solids]
        path = phase_wsolids.Liquid_1.path_data

        # TODO: Using phase_wsolids.getDensity() doesnn't allow updating fracs
        rho_liq = phase_wsolids.Liquid_1.getDensity(mass_frac=massfrac)
        rho_sol = phase_wsolids.Solid_1.getDensity()

        # Distribution balance
        vol_liq_in = mass_liquid / rho_liq
        vol_sol_in = mass_solid / rho_sol

        vol_in = vol_liq_in + vol_sol_in  # slurry volumes
        vol_total = sum(vol_in)

        vol_liq = total_liquid / rho_liq
        porosity = phase_wsolids.Solid_1.getPorosity()  # TODO: the inlet is not necessarily a Cake

        vol_pores = total_solid / rho_sol / ((1 - porosity) / porosity)
        if vol_liq > vol_pores:

            self.type_out = 'Slurry'
            distrib = distrib_in.sum(axis=0) / vol_total

            if self.is_continuous:
                self.Outlet = SlurryStream()

                liquid_out = LiquidStream(path, mass_flow=total_liquid,
                                          mass_frac=massfrac)
                solid_out = SolidStream(path, mass_flow=total_solid,
                                        mass_frac=phase_wsolids.Solid_1.mass_frac,
                                        distrib=distrib,
                                        x_distrib=phase_wsolids.Solid_1.x_distrib)
            else:
                liquid_out = LiquidPhase(path, mass=total_liquid,
                                         mass_frac=massfrac)
                solid_out = SolidPhase(path, mass=total_solid,
                                       mass_frac=phase_wsolids.Solid_1.mass_frac,
                                       distrib=distrib,
                                       x_distrib=phase_wsolids.Solid_1.x_distrib)
                self.Outlet = Slurry(vol_slurry=vol_total)

            self.Outlet.Phases = (liquid_out, solid_out)
        else:
            self.type_out = 'Cake'
            pass  # TODO: create Cake object

        # Energy balances
        temp_out = self.energy_balance(u_inputs)

        return total_liquid, total_solid, massfrac, distrib, temp_out

    def solve_unit(self):

        # ---------- Read inputs
        solids_flag = [inlet.__module__ == 'PharmaPy.MixedPhases'
                       for inlet in self.Inlets]

        if any(solids_flag):
            u_input, ind_solids = self.get_inputs_solids()

            path = self.Inlets[0].path_data
            self.Liquid_1 = LiquidPhase(path)
            if isinstance(u_input['mass_frac'], list):
                pass
            else:
                states = self.balances_solids(u_input, ind_solids)
        else:
            u_input = self.get_inputs()

            # ---------- Create output phase
            path = self.Inlets[0].path_data
            if self.is_continuous:
                self.Liquid_1 = LiquidStream(path_thermo=path)
            else:
                self.Liquid_1 = LiquidPhase(path_thermo=path)

            # ---------- Run balances
            if isinstance(u_input['mass_frac'], list):
                states = self.dynamic_balances(u_input)
            else:
                states = self.balances(u_input)

        # ---------- Retrieve results
        self.retrieve_results(states)

        return states

    def retrieve_results(self, states):
        solids_flag = [inlet.__module__ == 'PharmaPy.MixedPhases'
                       for inlet in self.Inlets]

        if any(solids_flag):
            mass_liq, mass_sol, massfrac_liq, distrib, temp = states

            if self.type_out == 'Slurry':
                self.names_states_out = ['mass_liq', 'temp', 'num_distrib']
            else:
                self.names_states_out = ['mass_liq', 'temp', 'total_distrib']

            self.outputs = states

            # Update phases
            self.Outlet.Liquid_1.updatePhase(mass=mass_liq,
                                             mass_frac=massfrac_liq)

            self.Outlet.Liquid_1.temp = temp

            self.Outlet.Solid_1.updatePhase(mass=mass_sol, distrib=distrib)

            self.Outlet.Solid_1.temp = temp

            self.timeProf = [0]

        else:
            mass, massfrac, temp = states
            # ---------- Update phases
            if massfrac.ndim == 1:
                last_massfrac = massfrac
                last_mass = mass

                self.outputs = np.hstack((massfrac, mass, temp))
            else:
                last_massfrac = massfrac[-1]
                last_mass = mass[-1]

                self.outputs = np.column_stack((massfrac, mass, temp))

            if self.is_continuous:
                self.Liquid_1.updatePhase(mass_frac=last_massfrac,
                                          mass_flow=last_mass)
            else:
                self.Liquid_1.temp = temp
                self.Liquid_1.updatePhase(mass_frac=last_massfrac,
                                          mass=last_mass)

            self.Outlet = self.Liquid_1
            self.outputs = np.atleast_2d(self.outputs)
            # self.timeProf = [0]


class DynamicCollector:
    def __init__(self, Inlet=None, timeshift_factor=1.5, temp_refer=298.15,
                 tau=None, name_species=None):

        self.Inlet = Inlet
        # if self.inlet is not None:
        #     classify_phases(self.inlet)

        self.tau = tau
        self.vol_offset = 0.75

        self.oper_mode = 'Dynamic'
        self.time_shift = timeshift_factor

        self._Phases = None

        self.is_continuous = False

        self.name_species = name_species
        self.num_species = len(name_species)
        self.names_upstream = None
        self.bipartite = None

        self.nomenclature()

        # Crystallizer instances
        self.KinCryst = None
        self.CrystInst = None
        self.is_cryst = None

        self.kwargs_cryst = None

        self.elapsed_time = 0

    def nomenclature(self):
        names_liquid = ['mass_frac', 'mass_flow', 'temp']
        names_solids = ['mass_conc', 'vol_flow', 'temp', 'num_distrib']

        self.name_idx = 0

        self.names_states_in = [names_liquid, names_solids]
        self.names_states_out = ['mass_frac', 'mass', 'temp']

    def get_inputs(self, time):
        all_inputs = self.Inlet.InterpolateInputs(time)

        if hasattr(self.Inlet, 'Solid_1'):
            num_distrib = self.Inlet.Solid_1.num_distrib
        else:
            num_distrib = 0
        inputs = get_dict_states(self.names_upstream, self.num_species,
                                 num_distrib, all_inputs)

        input_dict = {}
        for name in self.names_states_in[self.name_idx]:
            input_dict[name] = inputs[self.bipartite[name]]
        return input_dict

    def unit_model(self, time, states):
        # Calculate inlets
        u_values = self.get_inputs(time)

        fracs = states[:self.num_species]

        mass = states[self.num_species]
        temp = states[self.num_species + 1]

        material_balances = self.material_balances(time, fracs, mass, u_values)
        energy_balance = self.energy_balance(time, fracs, mass, temp, u_values)

        balances = np.append(material_balances, energy_balance)

        return balances

    def material_balances(self, time, fracs, mass, u_inputs):
        inlet_flow = u_inputs['mass_flow']
        inlet_fracs = u_inputs['mass_frac']

        dfrac_dt = inlet_flow / mass * (inlet_fracs - fracs)
        dm_dt = inlet_flow

        dmaterial_dt = np.append(dfrac_dt, dm_dt)

        return dmaterial_dt

    def energy_balance(self, time, fracs, mass, temp, u_inputs):
        inlet_flow = u_inputs['mass_flow']
        inlet_fracs = u_inputs['mass_frac']
        inlet_temp = u_inputs['temp']

        h_in = self.Inlet.getEnthalpy(temp=inlet_temp, mass_frac=inlet_fracs)
        h_tank = self.Liquid_1.getEnthalpy(temp=temp, mass_frac=fracs)
        cp_tank = self.Liquid_1.getCp(temp=temp, mass_frac=fracs)

        dtemp_dt = inlet_flow / mass / cp_tank * (h_in - h_tank)

        return dtemp_dt

    def solve_unit(self, runtime=None, time_grid=None, verbose=True):
        # Initial values
        init_dict = self.get_inputs(self.elapsed_time)
        temp_init = init_dict['temp']

        # self.name_species = self.Liquid_1.name_species
        # self.num_species = len(self.name_species)

        self.is_cryst = any('distr' in word for word in init_dict.keys())

        if self.is_cryst:
            path = self.Inlet.Liquid_1.path_data
            vol_init = init_dict['vol_flow'] / 100
            conc_init = init_dict['mass_conc']
            distr_init = init_dict['num_distrib'] * vol_init

            vol_init *= (1 - self.Inlet.moments[3])

            liquid = LiquidPhase(path, temp=temp_init, mass_conc=conc_init,
                                 vol=vol_init)

            frac_solid = np.zeros_like(conc_init)
            frac_solid[self.kwargs_cryst['target_ind']] = 1
            solid = SolidPhase(path, temp=temp_init, distrib=distr_init,
                               x_distrib=self.Inlet.Solid_1.x_distrib,
                               mass_frac=frac_solid)
            phases = (liquid, solid)

            SemiCryst = SemibatchCryst(method='1D-FVM', adiabatic=True,
                                       **self.kwargs_cryst)
            SemiCryst.Phases = phases
            SemiCryst.Kinetics = self.KinCryst
            SemiCryst.Inlet = self.Inlet

            SemiCryst.names_upstream = self.names_upstream
            SemiCryst.bipartite = self.bipartite

            SemiCryst.elapsed_time = self.elapsed_time

            time, states = SemiCryst.solve_unit(runtime, time_grid,
                                                verbose=verbose)

            # Retrieve crystallizer results
            output_names = ['timeProf', 'wConcProf', 'tempProf', 'Outlet',
                            'outputs']

            for name in output_names:
                setattr(self, name, getattr(SemiCryst, name))

            self.CrystInst = SemiCryst
            self.Outlet = SemiCryst.Outlet

            vol_phase = self.Outlet.vol_slurry
            if isinstance(vol_phase, np.ndarray):
                vol_phase = vol_phase[0]

            self.vol_phase = vol_phase
        else:
            path = self.Inlet.path_data
            mass_init = init_dict['mass_flow'] / 10
            frac_init = init_dict['mass_frac']

            liquid = LiquidPhase(path, temp=temp_init, mass_frac=frac_init)

            states_init = np.concatenate((frac_init, [mass_init, temp_init]))
            self.Phases = (liquid,)
            classify_phases(self)

            problem = Explicit_Problem(self.unit_model, states_init,
                                       t0=self.elapsed_time)
            solver = CVode(problem)

            if not verbose:
                solver.verbosity = 50

            if runtime is not None:
                final_time = runtime + self.elapsed_time

            if time_grid is not None:
                final_time = time_grid[-1]

            time, states = solver.simulate(final_time, ncp_list=time_grid)

            self.retrieve_results(time, states)

            vol_liq = self.Liquid_1.vol
            if isinstance(vol_liq, np.ndarray):
                vol_liq = vol_liq[0]

            self.vol_phase = vol_liq

        return time, states

    def retrieve_results(self, time, states):
        self.timeProf = np.array(time)
        self.elapsed_time = time[-1]

        self.wConcProf = states[:, :self.num_species]
        self.massProf = states[:, self.num_species]
        self.tempProf = states[:, self.num_species + 1]

        self.outputs = np.atleast_2d(states[-1])

        self.Liquid_1.updatePhase(mass_frac=self.wConcProf[-1],
                                  mass=self.massProf[-1])

        self.Liquid_1.temp = self.tempProf[-1]

        if self.is_cryst:
            self.Outlet = self.CrystInst.Outlet
        else:
            self.Outlet = self.Liquid_1

    def plot_profiles(self, fig_size=None, time_div=1, pick_comp=None,
                      kwargs=None):
        if kwargs is None:
            kwargs = {}

        if self.is_cryst:
            fig, axes, ax_right = self.CrystInst.plot_profiles(
                fig_size, time_div=time_div, **kwargs)
        else:
            fig, axes = self.plot_local(fig_size, time_div, pick_comp)

        return fig, axes

    def plot_local(self, fig_size=None, time_div=1, pick_comp=None):

        if pick_comp is None:
            pick_comp = range(self.wConcProf.shape[1])

        leg_comp = [self.name_species[ind] for ind in pick_comp]

        fig, axes = plt.subplots(2, 1, figsize=fig_size)

        # Mass fraction
        axes[0].plot(self.timeProf / time_div, self.wConcProf[:, pick_comp])

        axes[0].set_ylabel('mass frac')
        axes[0].legend(leg_comp)

        # Mass and temperature
        axes[1].plot(self.timeProf / time_div, self.massProf, 'k')
        axes[1].set_ylabel('mass (kg)')

        ax_temp = axes[1].twinx()
        ax_temp.plot(self.timeProf / time_div, self.tempProf, '--')
        ax_temp.set_ylabel('$T$ (K)')

        color = ax_temp.lines[0].get_color()
        ax_temp.spines['right'].set_color(color)
        ax_temp.tick_params(colors=color)
        ax_temp.yaxis.label.set_color(color)
        ax_temp.spines['top'].set_visible(False)

        for axis in axes:
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)

            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        if time_div == 1:
            fig.text(0.5, 0, 'time (s)', ha='center')

        return fig, axes
