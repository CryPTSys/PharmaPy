import numpy as np
from assimulo.problem import Implicit_Problem

from PharmaPy.Phases import classify_phases
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Commons import unpack_discretized, retrieve_pde_result
from PharmaPy.Results import DynamicResult
from PharmaPy.Streams import LiquidStream

from PharmaPy.Extractors import BatchExtractor

from assimulo.solvers import IDA

import copy


class DynamicExtractor:
    def __init__(self, num_stages, k_fun=None, gamma_model='UNIQUAC'):

        self.num_stages = num_stages

        if callable(k_fun):
           self.k_fun = k_fun
        else:
            pass  # Use UNIFAC/UNIQUAC

        self._Phases = None
        self._Inlet = None
        self.inlets = {}

        self.heavy_phase = None

    def nomenclature(self):
        num_comp = self.num_comp
        name_species = self.name_species
        self.states_di = {
            'mol_i': {'dim': num_comp, 'units': 'mole', 'type': 'diff', 'index': name_species},
            'x_i': {'dim': num_comp, 'type': 'alg', 'index': name_species},
            'y_i': {'dim': num_comp, 'type': 'alg', 'index': name_species},
            'holdup_R': {'dim': 1, 'type': 'alg', 'units': 'mole'},
            'holdup_E': {'dim': 1, 'type': 'alg', 'units': 'mole'},
            'R_flow': {'dim': 1, 'type': 'alg', 'units': 'mole/s'},
            'E_flow': {'dim': 1, 'type': 'alg', 'units': 'mole/s'},
            'u_int': {'dim': 1, 'type': 'diff', 'units': 'J'},
            'temp': {'dim': 1, 'type': 'alg', 'units': 'K'}
            }

        self.name_states = list(self.states_di.keys())
        self.dim_states = [di['dim'] for di in self.states_di.values()]

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if isinstance(phases, (list, tuple)):
            self._Phases = phases
        else:
            self._Phases = [phases]

        classify_phases(self)

        self.name_species = self.Liquid_1.name_species
        self.num_comp = len(self.name_species)

        self.nomenclature()

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):
        if isinstance(inlet, dict):
            if 'raffinate' not in inlet.keys() and 'extract' not in inlet.keys():
                raise KeyError(
                    "The passed dictionary must have the key 'raffinate' "
                    "or the key 'extract' identified the passed streams ")

        else:
            raise TypeError(
                "'inlet' object must be a dictionary containing one or "
                "both 'raffinate' or 'extract' as keys and "
                "LiquidStreams as values")

        self.inlets = inlet | self.inlets
        self._Inlet = self.inlets

    def get_inputs(self, time):
        inputs = get_inputs_new(time, self.Inlet, self.states_in_di)

        return inputs

    def unit_model(self, time, states):

        di_states = unpack_discretized(states,
                                       self.dim_states, self.name_states)

        material = self.material_balances(time, di_states)
        energy = self.energy_balances(time, **di_states)

        balances = np.hstack(material, energy)
        return balances

    def material_balances(self, time, mol_i, x_i, y_i,
                          holdup_R, holdup_E, R_flow, E_flow, u_int, temp):

        return

    def energy_balances(self, time, mol_i, x_i, y_i,
                        holdup_R, holdup_E, R_flow, E_flow, u_int, temp):

        return

    def initialize_model(self):
        extr = BatchExtractor(k_fun=self.k_fun)
        extr.Phases = copy.deepcopy(self.Liquid_1)

        extr.solve_unit()
        res = extr.result

        mol_i_init = res.x_heavy * res.mol_heavy + res.x_light * res.mol_light

        rhos_streams = {key: obj.getDensity()
                        for key, obj in self.Inlet.items()}

        rhos_holdups = [res.rho_heavy, res.rho_light]

        idx_raff = np.argmin(abs(rhos_holdups - rhos_streams['raffinate']))

        if idx_raff == 0:
            target_states = {'x_i': 'x_heavy', 'y_i': 'x_light',
                             'holdup_R': 'mol_heavy', 'holdup_E': 'mol_light'}

            self.heavy_phase = 'raffinate'
        elif idx_raff == 1:
            target_states = {'x_i': 'x_light', 'y_i': 'x_heavy',
                             'holdup_R': 'mol_light', 'holdup_E': 'mol_heavy'}

            self.heavy_phase = 'extract'

        di_init = {'mol_i': mol_i_init}

        for name, equiv in target_states.items():
            di_init[name] = getattr(res, equiv)

        return di_init

    def solve_unit(self, runtime):

        temp_init = self.Liquid_1.temp
        di_init = self.initialize_model()

        # init_states = np.tile(np.hstack(list(di_init.values())),
        #                       (self.num_plates + 1, 1))

        # init_states = np.hstack(())
        # init_states = init_states.T.ravel()

        # problem = Implicit_Problem(self.unit_model)
        # solver = IDA(problem)

        # time, states = solver.solve(runtime)

        # return time, states

        return di_init

    def retrieve_results(self, time, states):
        time = np.asarray(time)

        di = unpack_discretized(states, self.dim_states, self.name_states)
        di['stage'] = np.arange(1, self.num_stages + 1)
        di['time'] = time

        self.result = DynamicResult(self.states_di, self.fstates_di, **di)

        di_last = retrieve_pde_result(di, 'stage',
                                      time=time[-1], x=self.num_stages)

        self.Outlet = LiquidStream(self.Liquid_1.path_thermo,
                                   temp=di_last['temp'],
                                   mole_flow=di['mole_flow'])
