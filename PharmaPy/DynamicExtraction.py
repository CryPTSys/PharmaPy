import numpy as np
from scipy import linalg
from assimulo.problem import Implicit_Problem

from PharmaPy.Phases import classify_phases
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Commons import unpack_discretized, retrieve_pde_result
from PharmaPy.Results import DynamicResult
from PharmaPy.Streams import LiquidStream

from PharmaPy.Extractors import BatchExtractor
from PharmaPy.Plotting import plot_distrib

from assimulo.solvers import IDA
from assimulo.solvers import Radau5DAE

import copy

from matplotlib.ticker import MaxNLocator, AutoMinorLocator


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
            'holdup_light': {'dim': 1, 'type': 'alg', 'units': 'mole'},
            'holdup_heavy': {'dim': 1, 'type': 'alg', 'units': 'mole'},
            # 'top_flows': {'dim': 1, 'type': 'alg', 'units': 'mole/s'},
            'u_int': {'dim': 1, 'type': 'diff', 'units': 'J'},
            'temp': {'dim': 1, 'type': 'alg', 'units': 'K'}
            }

        self.name_states = list(self.states_di.keys())
        self.dim_states = [di['dim'] for di in self.states_di.values()]

        states_in_dict = {'mole_flow': 1, 'temp': 1, 'mole_frac': num_comp}
        self.states_in_dict = {'Inlet': states_in_dict}

        self.alg_map = get_alg_map(self.states_di, self.num_stages)

        self.fstates_di = {}

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
            if 'feed' not in inlet.keys() and 'solvent' not in inlet.keys():
                raise KeyError(
                    "The passed dictionary must have the key 'feed' "
                    "or the key 'solvent' identified the passed streams ")

        else:
            raise TypeError(
                "'inlet' object must be a dictionary containing one or "
                "both 'feed' or 'solvent' as keys and "
                "LiquidStreams as values")

        self.inlets = inlet | self.inlets
        self._Inlet = self.inlets

    def get_stage_dimensions(self, di_states, rhos):
        vol = di_states['holdup_heavy'][0] / rhos[0] + \
            di_states['holdup_light'][0] / rhos[1]

        self.vol = vol / 1000  # m**3
        if self.area_cross is None:
            diam_stage = (4 * self.dh_ratio * self.vol / np.pi)**(1/3)
            self.area_cross = np.pi/4 * diam_stage**2

            self.diam_stage = diam_stage

    def get_inputs(self, time):
        inputs = {key: get_inputs_new(time, inlet, self.states_in_dict) for
                  key, inlet in self.Inlet.items()}

        return inputs

    # def get_bottom_flows(self, di_states, rhos, mws):
    #     rho_mass = rhos[0] * mws[0]  # kg/m**3

    #     vel_out = np.sqrt(2 * 9.81 / rho_mass / self.area_cross *
    #                       (mws[0] * di_states['holdup_heavy'] +
    #                        mws[1] * di_states['holdup_light']) / 1000
    #                       )  # m/s

    #     bottom_flows = self.cd * rhos[0] * self.area_out * vel_out  # * 1000

    #     return bottom_flows

    # def get_top_flow_arrays(self, bottom_flows, light_flow, rhos):
    #     rng = np.arange(self.num_stages - 1)

    #     rho_heavy, rho_light = rhos

    #     a_matrix = -np.eye(self.num_stages) * 1/rho_light
    #     a_matrix[rng + 1, rng] = 1/rho_light[:-1]

    #     b_vector = -1 / rho_heavy * (bottom_flows[1:] - bottom_flows[:-1])
    #     b_vector[0] -= light_flow / rho_light[0]

    #     return a_matrix, b_vector

    def get_augmented_arrays(self, di_states, inputs):  # bottom_flows):
        light = self.target_states['light_phase']
        heavy = self.target_states['heavy_phase']

        x_in = inputs[light]['Inlet']['mole_frac']
        y_in = inputs[heavy]['Inlet']['mole_frac']

        temp_in = {key: val['Inlet']['temp'] for key, val in inputs.items()}

        x_augm = np.vstack((x_in, di_states['x_i']))
        y_augm = np.vstack((di_states['y_i'], y_in))
        temp_augm = np.hstack((temp_in['feed'], di_states['temp'],
                               temp_in['solvent']))

        light_flows = np.zeros(self.num_stages + 1)
        heavy_flows = np.zeros_like(light_flows)

        heavy_in = inputs[heavy]['Inlet']['mole_flow']
        light_in = inputs[light]['Inlet']['mole_flow']

        light_flows[0] = light_in
        # light_flows[1:] = di_states['top_flows']
        light_flows[1:] = light_in

        heavy_flows[-1] = heavy_in
        # heavy_flows[:-1] = bottom_flows
        heavy_flows[:-1] = heavy_in

        augm_arrays = (x_augm, y_augm, temp_augm, light_flows, heavy_flows)

        return augm_arrays

    def unit_model(self, time, states, sdot=None):

        # ---------- Unpack variables
        di_states = unpack_discretized(states,
                                       self.dim_states, self.name_states)

        if sdot is None:
            di_sdot = sdot
        else:
            di_sdot = unpack_discretized(sdot, self.dim_states,
                                         self.name_states)

            # print('\ntime: ', time, end='\n')
            # fields = ('holdup_light', 'holdup_heavy')
            # di_print = {key: di_states[key] for key in fields}

            # print(di_print)

            # fields = ('mol_i', )
            # di_print = {key: {'max': di_sdot[key].max(),
            #                   'comp': self.name_species[np.argmax(di_sdot[key])]} for key in fields}

            # print(di_print)

        inputs = self.get_inputs(time)

        keys_frac = ('y_i', 'x_i')
        mws = [self.Liquid_1.getMolWeight(mole_frac=di_states[key])
               for key in keys_frac]

        rhos = [
            self.Liquid_1.getDensity(mole_frac=di_states[key], basis='mole',
                                     temp=di_states['temp'])
            for key in keys_frac]

        # bottom_flows = self.get_bottom_flows(di_states, rhos, mws)

        augm_arrays = self.get_augmented_arrays(di_states, inputs)
                                                # bottom_flows)

        # ---------- Balances
        material = self.material_balances(time,
                                          augm_arrays=augm_arrays,
                                          di_sdot=di_sdot, rhos=rhos,
                                          **di_states)

        energy = self.energy_balances(time,
                                      augm_arrays=augm_arrays,
                                      di_sdot=di_sdot,
                                      **di_states)

        balances = np.column_stack(material + energy).ravel()

        return balances

    def material_balances(self, time, mol_i, x_i, y_i,
                          holdup_light, holdup_heavy,  # top_flows,
                          u_int, temp,
                          di_sdot, rhos, augm_arrays):

        x_augm, y_augm, temp_augm, light_flows, heavy_flows = augm_arrays

        rho_heavy, rho_light = rhos

        # ---------- Differential block
        dnij_dt = y_augm[1:] * heavy_flows[1:, np.newaxis] \
            + x_augm[:-1] * light_flows[:-1, np.newaxis] \
            - y_augm[:-1] * heavy_flows[:-1, np.newaxis] \
            - x_augm[1:] * light_flows[1:, np.newaxis]

        if di_sdot is not None:
            dnij_dt = dnij_dt - di_sdot['mol_i']

        # ---------- Algebraic block
        nij_alg = x_i * holdup_light[:, np.newaxis] \
            + y_i * holdup_heavy[:, np.newaxis] - mol_i

        k_ij = self.k_fun(x_i, y_i, temp)  # TODO: make this stage-wise
        equilibrium_alg = k_ij * x_i - y_i

        global_alg = holdup_light + holdup_heavy - mol_i.sum(axis=1)
        volume_alg = holdup_light/rho_light + holdup_heavy/rho_heavy \
            - self.vol * 1000

        # top_flow_alg = (light_flows[:-1] - light_flows[1:]) / rho_light \
        #     + (heavy_flows[1:] - heavy_flows[:-1]) / rho_heavy

        out = [dnij_dt, nij_alg, equilibrium_alg, global_alg, volume_alg]
               # top_flow_alg]

        # di_flows = {'heavy': {'in': heavy_flows[1], 'out': heavy_flows[0]},
        #             'light': {'in': light_flows[0], 'out': light_flows[1]}}

        # print(di_flows)

        return out

    def energy_balances(self, time, mol_i, x_i, y_i,
                        holdup_light, holdup_heavy,  # top_flows,
                        u_int, temp,
                        di_sdot, augm_arrays):

        x_augm, y_augm, temp_augm, light_flows, heavy_flows = augm_arrays

        h_light = self.Liquid_1.getEnthalpy(mole_frac=x_augm,
                                            temp=temp_augm[:-1],
                                            basis='mole')

        h_heavy = self.Liquid_1.getEnthalpy(mole_frac=y_augm,
                                            temp=temp_augm[1:],
                                            basis='mole')

        duint_dt = heavy_flows[1:] * h_heavy[1:] \
            + light_flows[:-1] * h_light[:-1] \
            - heavy_flows[:-1] * h_heavy[:-1] - light_flows[1:] * h_light[1:]

        if di_sdot is not None:
            duint_dt = duint_dt - di_sdot['u_int']

        temp_eqns = holdup_heavy * h_heavy[:-1] + holdup_light * h_light[1:] \
            - u_int

        out = [duint_dt, temp_eqns]

        return out

    def initialize_model(self):
        # ---------- Equilibrium calculations
        extr = BatchExtractor(k_fun=self.k_fun)
        extr.Phases = copy.deepcopy(self.Liquid_1)

        extr.solve_unit()
        res = extr.result

        mol_i_init = res.x_heavy * res.mol_heavy + res.x_light * res.mol_light

        # ---------- Discriminate heavy and light phases
        rhos_streams = {key: obj.getDensity(basis='mole')
                        for key, obj in self.Inlet.items()}

        rhos_holdups = [res.rho_heavy, res.rho_light]

        idx_raff = np.argmin(abs(rhos_holdups - rhos_streams['feed']))

        target_states = {'x_light': 'x_i', 'x_heavy': 'y_i',
                         'mol_light': 'holdup_light',
                         'mol_heavy': 'holdup_heavy'}

        if idx_raff == 0:
            target_states['light_phase'] = 'solvent'
            target_states['heavy_phase'] = 'feed'

        elif idx_raff == 1:
            target_states['light_phase'] = 'feed'
            target_states['heavy_phase'] = 'solvent'

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

        di_init['temp'] = self.Liquid_1.temp * np.ones(self.num_stages)

        keys_frac = ('y_i', 'x_i')
        mw_holdups = [self.Liquid_1.getMolWeight(mole_frac=di_init[key])
                      for key in keys_frac]

        rhos_holdups = [
            self.Liquid_1.getDensity(mole_frac=di_init[key], basis='mole',
                                     temp=di_init['temp'])
            for key in keys_frac]

        # Get flows
        inputs = self.get_inputs(0)

        self.get_stage_dimensions(di_init,
                                  [rhos_holdups[0][0], rhos_holdups[1][0]])

        # bottom_flows = self.get_bottom_flows(di_init, rhos_holdups, mw_holdups)

        # bottom_holder = np.zeros(self.num_stages + 1)

        heavy = target_states['heavy_phase']
        light = target_states['light_phase']

        heavy_in = inputs[heavy]['Inlet']['mole_flow']
        light_in = inputs[light]['Inlet']['mole_flow']

        # bottom_holder[-1] = heavy_in
        # bottom_holder[:-1] = bottom_flows

        # top_flow_arrays = self.get_top_flow_matrix(bottom_holder, light_in,
        #                                            rhos_holdups)

        # top_flows = linalg.solve(*top_flow_arrays)

        # di_init['top_flows'] = top_flows

        # Energy balance calculations
        h_light = self.Liquid_1.getEnthalpy(mole_frac=di_init['x_i'],
                                            temp=di_init['temp'],
                                            basis='mole')

        h_heavy = self.Liquid_1.getEnthalpy(mole_frac=di_init['y_i'],
                                            temp=di_init['temp'],
                                            basis='mole')

        u_int = di_init['holdup_light'] * h_light \
            + di_init['holdup_heavy'] * h_heavy

        di_init['u_int'] = u_int

        return di_init

    def solve_unit(self, runtime, sundials_opts=None):

        di_init = self.initialize_model()
        di_init = {key: di_init[key] for key in self.name_states}

        init_states = np.column_stack(tuple(di_init.values()))
        init_states = init_states.ravel()

        init_deriv = self.unit_model(0, init_states)

        problem = Implicit_Problem(self.unit_model,
                                   y0=init_states, yd0=init_deriv)

        problem.algvar = self.alg_map

        solver = Radau5DAE(problem)
        # solver.make_consistent('IDA_YA_YDP_INIT')
        # solver.suppress_alg = True

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        time, states, sdot = solver.simulate(runtime)

        self.retrieve_results(time, states)

        return time, states

    def retrieve_results(self, time, states):
        time = np.asarray(time)

        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        di = unpack_discretized(states, self.dim_states, self.name_states,
                                indexes=indexes)
        di['stage'] = np.arange(1, self.num_stages + 1)
        di['time'] = time

        self.result = DynamicResult(self.states_di, self.fstates_di, **di)

        # di_last = retrieve_pde_result(di, 'stage',
        #                               time=time[-1], x=self.num_stages)

        # self.Outlet = LiquidStream(self.Liquid_1.path_thermo,
        #                            temp=di_last['temp'],
        #                            mole_flow=di['mole_flow'])

    def plot_profiles(self, times=None, stages=None, pick_comp=None,
                      **fig_kwargs):

        if pick_comp is None:
            states_plot = ('holdup_light', 'holdup_heavy', 'x_i', 'y_i')
        else:
            states_plot = ('holdup_light', 'holdup_heavy',
                           ('x_i', pick_comp), ('y_i', pick_comp))

        ylabels = ('H_light', 'H_heavy', 'x_i', 'y_i')

        fig, axis = plot_distrib(self, states_plot, 'stage', times=times,
                                 x_vals=stages, nrows=2, ncols=2,
                                 ylabels=ylabels, **fig_kwargs)

        if times is not None:
            fig.text(0.5, 0, 'stage', ha='center')

            for ax in axis:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        elif stages is not None:
            fig.text(0.5, 0, '$t$ (s)', ha='center')

            for ax in axis:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        for ax in axis:
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        fig.tight_layout()

        return fig, axis

