import numpy as np
from assimulo.problem import Implicit_Problem

from PharmaPy.Phases import classify_phases
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Commons import unpack_discretized, retrieve_pde_result
from PharmaPy.Results import DynamicResult
from PharmaPy.Streams import LiquidStream

from assimulo.solvers import IDA


class DynamicExtractor:
    def __init__(self, ):

        self.nomenclature()
        self._Phases = None
        self._Inlet = None

    def nomenclature(self):
        num_comp = self.num_comp
        name_species = self.name_species
        self.states_di = {
            'mol_i': {'dim': num_comp, 'units': 'mole', 'type': 'diff', 'index': name_species},
            'x_liq': {'dim': num_comp, 'type': 'alg', 'index': name_species},
            'y_liq': {'dim': num_comp, 'type': 'alg', 'index': name_species},
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
        self.num_species = len(self.name_species)

        self.nomenclature()

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):

        classify_phases(self)

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

    def material_balances(self, time, mol_i, x_liq, y_liq,
                          holdup_R, holdup_E, R_flow, E_flow, u_int, temp):

        return

    def energy_balances(self, time, mol_i, x_liq, y_liq,
                        holdup_R, holdup_E, R_flow, E_flow, u_int, temp):

        return

    def solve_model(self, runtime):

        init_states = np.hstack(())
        init_states = init_states.T.ravel()

        problem = Implicit_Problem(self.unit_model)
        solver = IDA(problem)

        time, states = solver.solve(runtime)

        return time, states

    def retrieve_results(self, time, states):
        time = np.asarray(time)

        di = unpack_discretized(states, self.dim_states, self.name_states)
        di['stage'] = np.arange(1, self.num_stages + 1)
        di['time'] = time

        result = DynamicResult(self.states_di, self.fstates_di, **di)

        di_last = retrieve_pde_result(di, 'stage',
                                      time=time[-1], x=self.num_stages)

        self.Outlet = LiquidStream(self.Liquid_1.path_thermo,
                                   temp=di_last['temp'],
                                   mole_flow=di['mole_flow'])
