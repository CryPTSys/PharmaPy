import numpy as np
from scipy import linalg
from assimulo.problem import Implicit_Problem

from PharmaPy.Phases import classify_phases
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Commons import unpack_discretized, retrieve_pde_result
from PharmaPy.Results import DynamicResult
from PharmaPy.Streams import LiquidStream

from PharmaPy.Extractors import BatchExtractor

from assimulo.solvers import IDA

import copy


def get_alg_map(states_di, nstages=1):
    maps = []
    for val in states_di.values():
        if val['type'] == 'diff':
            maps.append(np.ones(val['dim'] * nstages))
        elif val['type'] == 'alg':
            maps.append(np.zeros(val['dim'] * nstages))

    return np.hstack(maps)


class DynamicExtractor:
    def __init__(self, num_stages, k_fun=None, gamma_model='UNIQUAC',
                 area_cross=None, coeff_disch=1, diam_out=0.0254, dh_ratio=2):

        self.num_stages = num_stages

        if callable(k_fun):
           self.k_fun = k_fun
        else:
            pass  # Use UNIFAC/UNIQUAC

        self._Phases = None
        self._Inlet = None
        self.inlets = {}

        self.heavy_phase = None

        self.area_cross = area_cross
        self.diam_stage = None
        self.dh_ratio = dh_ratio

        self.area_out = np.pi / 4 * diam_out**2
        self.cd = coeff_disch

    def nomenclature(self):
        num_comp = self.num_comp
        name_species = self.name_species
        self.states_di = {
            'mol_i': {'dim': num_comp, 'units': 'mole', 'type': 'diff',
                      'index': name_species},
            'x_i': {'dim': num_comp, 'type': 'alg', 'index': name_species},
            'y_i': {'dim': num_comp, 'type': 'alg', 'index': name_species},
            'holdup_R': {'dim': 1, 'type': 'alg', 'units': 'mole'},
            'holdup_E': {'dim': 1, 'type': 'alg', 'units': 'mole'},
            'top_flows': {'dim': 1, 'type': 'alg', 'units': 'mole/s'},
            'u_int': {'dim': 1, 'type': 'diff', 'units': 'J'},
            'temp': {'dim': 1, 'type': 'alg', 'units': 'K'}
            }

        self.name_states = list(self.states_di.keys())
        self.dim_states = [di['dim'] for di in self.states_di.values()]

        states_in_dict = {'mole_flow': 1, 'temp': 1, 'mole_frac': num_comp}
        self.states_in_dict = {'Inlet': states_in_dict}

        self.alg_map = get_alg_map(self.states_di, self.num_stages)

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

    def get_stage_dimensions(self, diph):
        self.vol = diph['heavy']['vol'] + diph['light']['vol']
        if self.area_cross is None:
            diam_stage = (4 * self.dh_ratio * self.vol / np.pi)**(1/3)
            self.area_cross = np.pi/4 * diam_stage**2

            self.diam_stage = diam_stage

    def get_inputs(self, time):
        inputs = {key: get_inputs_new(time, inlet, self.states_in_dict) for
                  key, inlet in self.Inlet.items()}

        return inputs

    def get_bottom_flows(self, diph):
        rho_mass = diph['heavy']['rho'] * diph['heavy']['mw']
        bottom_flows = self.cd * diph['heavy']['rho'] * self.area_out * \
            np.sqrt(2 * 9.18 / rho_mass / self.area_cross *
                    (diph['heavy']['mw'] * diph['heavy']['mol'] +
                     diph['light']['mw'] * diph['light']['mol']) / 1000
                    )

        return bottom_flows

    def get_top_flow_eqns(self, top_flows, bottom_flows, diph,
                          matrix=False):

        if matrix:
            rng = np.arange(self.num_stages - 1)
            a_matrix = -np.eye(self.num_stages) * 1/diph['light']['rho']

            if self.target_states['heavy_phase'] == 'raffinate':
                a_matrix[rng, rng + 1] = 1/diph['light']['rho'][1:]

                b_vector = 1 / diph['heavy']['rho'] \
                    * (bottom_flows[1:] - bottom_flows[:-1])

                b_vector[-1] -= top_flows[-1]/diph['light']['rho'][-1]

            elif self.target_states['heavy_phase'] == 'extract':
                a_matrix[rng + 1, rng] = 1/diph['light']['rho'][:-1]

                b_vector = -1 / diph['heavy']['rho'] \
                    * (bottom_flows[1:] - bottom_flows[:-1])
                b_vector[0] -= top_flows[0]/diph['light']['rho'][0]

            eqns = (a_matrix, b_vector)
        else:
            if self.target_states['heavy_phase'] == 'raffinate':  # TODO: check
                eqns = (top_flows[1:] - top_flows[:-1]) / diph['light']['rho'] \
                    + (bottom_flows[:-1] - bottom_flows[1:]) / diph['heavy']['rho']
            elif self.target_states['heavy_phase'] == 'extract':
                eqns = (top_flows[:-1] - top_flows[1:]) / diph['light']['rho'] \
                    + (bottom_flows[1:] - bottom_flows[:-1]) / diph['heavy']['rho']

        return eqns

    def get_u_int(self, di_states):
        h_R = self.Liquid_1.getEnthalpy(mole_frac=di_states['x_i'],
                                        temp=di_states['temp'])

        h_E = self.Liquid_1.getEnthalpy(mole_frac=di_states['y_i'],
                                        temp=di_states['temp'])

        u_int = di_states['holdup_R'] * h_R + di_states['holdup_E'] * h_E

        return u_int

    def unit_model(self, time, states):

        # ---------- Unpack variables
        di_states = unpack_discretized(states,
                                       self.dim_states, self.name_states)

        inputs = self.get_inputs(time)

        diph = self.get_di_phases(di_states)
        bottom_flows = self.get_bottom_flows(diph)

        x_in = inputs['raffinate']['Inlet']['mole_frac']
        y_in = inputs['extract']['Inlet']['mole_frac']

        # ---------- Create augmented arrays to account for inlets
        x_augm = np.vstack((x_in, di_states['x_i']))
        y_augm = np.vstack((di_states['y_i'], y_in))

        R_flows = np.zeros(self.num_stages + 1)
        E_flows = np.zeros_like(R_flows)

        rho_R = np.zeros_like(R_flows)
        rho_E = np.zeros_like(R_flows)
        if self.target_states['heavy_phase'] == 'raffinate':
            # Extract (light)
            E_flows[-1] = inputs['extract']['Inlet']['mole_flow']
            E_flows[:-1] = di_states['top_flows']

            rho_E[-1] = self.Liquid_1.getDensity(
                mole_frac=inputs['extract']['Inlet']['mole_frac'],
                temp=inputs['extract']['Inlet']['temp'])
            rho_E[:-1] = diph['light']['rho']

            # Raffinate (heavy)
            R_flows[0] = inputs['raffinate']['Inlet']['mole_flow']
            R_flows[1:] = bottom_flows

            rho_R[0] = self.Liquid_1.getDensity(
                mole_frac=inputs['raffinate']['Inlet']['mole_frac'],
                temp=inputs['raffinate']['Inlet']['temp'])
            rho_R[1:] = diph['heavy']['rho']

        elif self.target_states['heavy_phase'] == 'extract':
            # Raffinate (light)
            R_flows[0] = inputs['raffinate']['Inlet']['mole_flow']
            R_flows[1:] = di_states['top_flows']

            rho_R[0] = self.Liquid_1.getDensity(
                mole_frac=inputs['raffinate']['Inlet']['mole_frac'],
                temp=inputs['raffinate']['Inlet']['temp'])
            rho_R[1:] = diph['light']['rho']

            # Extract (heavy)
            E_flows[-1] = inputs['extract']['Inlet']['mole_flow']
            E_flows[:-1] = bottom_flows

            rho_E[-1] = self.Liquid_1.getDensity(
                mole_frac=inputs['extract']['Inlet']['mole_frac'],
                temp=inputs['extract']['Inlet']['temp'])
            rho_E[:-1] = diph['heavy']['rho']

        # Assign augmented fracs
        augm_arrays = (x_augm, y_augm, R_flows, E_flows, rho_R, rho_E)

        # ---------- Balances
        material = self.material_balances(time,
                                          augm_arrays=augm_arrays,
                                          di_phases=diph,
                                          **di_states)
        # energy = self.energy_balances(time, heavy_phase=heavy_phase,
        #                               **di_states)

        # balances = np.hstack((material, energy))
        return material

    def get_di_phases(self, di_states):
        di_phases = {'heavy': {}, 'light': {}}

        target_states = self.target_states.copy()
        target_states.pop('heavy_phase')
        for equiv, name in target_states.items():

            key = equiv.split('_')[0]
            if 'heavy' in equiv:
                phase = 'heavy'

            elif 'light' in equiv:
                phase = 'light'

            di_phases[phase][key] = di_states[name]

            if 'x_' in name or 'y_' in name:
                rho = self.Liquid_1.getDensity(mole_frac=di_states[name], basis='mole')
                mw = np.dot(self.Liquid_1.mw, di_states[name].T)

                di_phases[phase]['rho'] = rho
                di_phases[phase]['mw'] = mw

        di_phases['heavy']['vol'] = di_phases['heavy']['mol'] / di_phases['heavy']['rho']
        di_phases['light']['vol'] = di_phases['light']['mol'] / di_phases['light']['rho']

        return di_phases

    def material_balances(self, time, mol_i, x_i, y_i,
                          holdup_R, holdup_E, top_flows, u_int, temp,
                          di_phases, augm_arrays):

        x_augm, y_augm, R_flows, E_flows, rho_R, rho_E = augm_arrays

        # ---------- Differential block
        dnij_dt = y_augm[1:] * E_flows[1:, np.newaxis] + \
            x_augm[:-1] * R_flows[:-1, np.newaxis] - \
            y_augm[:-1] * E_flows[:-1, np.newaxis] + \
            x_augm[1:] * R_flows[1:, np.newaxis]

        # ---------- Algebraic block
        nij_alg = x_i * holdup_R[:, np.newaxis] + y_i * holdup_E[:, np.newaxis] \
            - mol_i

        k_ij = self.k_fun(x_i, y_i, temp)  # TODO: make this stage-wise
        equilibrium_alg = k_ij * x_i - y_i

        global_alg = holdup_R + holdup_E - mol_i.sum(axis=1)
        volume_alg = holdup_R/rho_R[1:] + holdup_E/rho_E[:-1] - self.vol

        if self.target_states['heavy_phase'] == 'raffinate':
            bottom_flows = R_flows
            top_augm = E_flows
        elif self.target_states['heavy_phase'] == 'extract':
            bottom_flows = E_flows
            top_augm = R_flows

        top_flow_alg = self.get_top_flow_eqns(top_augm, bottom_flows,
                                              di_phases)

        out = np.column_stack((dnij_dt,
                               nij_alg,  # good
                               equilibrium_alg,  # good
                               global_alg,  # check
                               volume_alg,  # good
                               top_flow_alg)  # good
                              )

        return out

    def energy_balances(self, time, mol_i, x_i, y_i,
                        holdup_R, holdup_E, top_flows, u_int, temp,
                        heavy_phase, bottom_flows):

        return

    def initialize_model(self):
        # ---------- Equilibrium calculations
        extr = BatchExtractor(k_fun=self.k_fun)
        extr.Phases = copy.deepcopy(self.Liquid_1)

        extr.solve_unit()
        res = extr.result

        mol_i_init = res.x_heavy * res.mol_heavy + res.x_light * res.mol_light

        # ---------- Discriminate heavy and light phases
        rhos_streams = {key: obj.getDensity()
                        for key, obj in self.Inlet.items()}

        rhos_holdups = [res.rho_heavy, res.rho_light]

        idx_raff = np.argmin(abs(rhos_holdups - rhos_streams['raffinate']))

        if idx_raff == 0:
            target_states = {'x_heavy': 'x_i', 'x_light': 'y_i',
                             'mol_heavy': 'holdup_R', 'mol_light': 'holdup_E',
                             'heavy_phase': 'raffinate'}

        elif idx_raff == 1:
            target_states = {'x_light': 'x_i', 'x_heavy': 'y_i',
                             'mol_light': 'holdup_R', 'mol_heavy': 'holdup_E',
                             'heavy_phase': 'extract'}

        self.target_states = target_states

        # ---------- Create dictionary of initial values
        di_init = {'mol_i': np.tile(mol_i_init, (self.num_stages, 1))}

        for equiv, name in target_states.items():
            if 'phase' not in equiv:
                attr = getattr(res, equiv)
                if isinstance(attr, np.ndarray):
                    if attr.ndim > 0:
                        attr = np.tile(attr, (self.num_stages, 1))
                    else:
                        attr = np.repeat(attr[0], self.num_stages)
                else:
                    attr = np.ones(self.num_stages) * attr

                di_init[name] = attr

        # Get flows and internal energies
        inputs = self.get_inputs(0)

        di_phases = self.get_di_phases(di_init)
        self.get_stage_dimensions(di_phases)

        bottom_flows = self.get_bottom_flows(di_phases)

        bottom_holder = np.zeros(self.num_stages + 1)
        top_holder = np.zeros_like(bottom_holder)

        if target_states['heavy_phase'] == 'raffinate':
            # Extract at the end of top_flows
            top_holder[-1] = inputs['extract']['Inlet']['mole_flow']

            # Raffinate at the beginning of bottom_flows
            bottom_holder[0] = inputs['raffinate']['Inlet']['mole_flow']
            bottom_holder[1:] = bottom_flows

        elif target_states['heavy_phase'] == 'extract':
            # Extract ath the end of bottom_flows
            bottom_holder[-1] = inputs['extract']['Inlet']['mole_flow']
            bottom_holder[:-1] = bottom_flows

            # Raffinate at the beginning of top_flows
            top_holder[0] = inputs['raffinate']['Inlet']['mole_flow']

        top_flow_eqns = self.get_top_flow_eqns(top_holder, bottom_holder,
                                               di_phases, matrix=True)

        top_flows = linalg.solve(*top_flow_eqns)

        di_init['top_flows'] = top_flows

        di_init['temp'] = self.Liquid_1.temp * np.ones(self.num_stages)

        u_int = self.get_u_int(di_init)

        di_init['u_int'] = u_int

        return di_init

    def solve_unit(self, runtime):

        di_init = self.initialize_model()
        di_init = {key: di_init[key] for key in self.name_states}

        init_states = np.column_stack(tuple(di_init.values()))
        init_states = init_states.ravel()

        balances = self.unit_model(0, init_states)

        # problem = Implicit_Problem(self.unit_model)
        # solver = IDA(problem)

        # time, states = solver.solve(runtime)

        # return time, states

        # return di_init, init_states

        return balances

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
