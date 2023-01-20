import numpy as np
from scipy import linalg
from scipy.optimize import root
from assimulo.problem import Implicit_Problem

from PharmaPy.Phases import classify_phases
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Commons import unpack_discretized, retrieve_pde_result, flatten_states
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


def complete_molefrac(mole_frac, mapping):
    if mole_frac.ndim == 1:
        out = np.zeros(len(mole_frac) + 1)
        out[mapping] = mole_frac
        out[~mapping] = 1 - sum(mole_frac)
    else:
        out = np.zeros((mole_frac.shape[0], mole_frac.shape[1] + 1))
        out[:, mapping] = mole_frac
        out[:, ~mapping] = 1 - mole_frac.sum(axis=1)

    return out


class DynamicExtractor:
    def __init__(self, num_stages, k_fun=None, eff=1, gamma_model='UNIQUAC'):

        self.num_stages = num_stages

        if callable(k_fun):
            self.k_fun = k_fun
        else:
            pass  # Use UNIFAC/UNIQUAC

        self._Phases = None
        self._Inlet = None
        self.inlets = {}

        self.names_states_in = ('mole_flow', 'temp', 'mole_frac')

        self.heavy_phase = None
        self.eff = eff

        self.fixed_vals = {}
        # self.stream_out = stream_out

        self.profiles_runs = []
        self.oper_mode = 'Continuous'
        self.is_continuous = True
        self.default_output = 'feed'

    def nomenclature(self):
        num_comp = self.num_comp
        name_species = self.name_species
        self.states_di = {
            # 'mol_i': {'dim': num_comp, 'units': 'mole', 'type': 'diff',
            #           'index': name_species},
            'x_i': {'dim': num_comp, 'type': 'alg', 'index': name_species},
            'y_i': {'dim': num_comp, 'type': 'alg', 'index': name_species},
            # 'holdup_light': {'dim': 1, 'type': 'alg', 'units': 'mole'},
            # 'holdup_heavy': {'dim': 1, 'type': 'alg', 'units': 'mole'},
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

        self.names_states_out = ('mole_frac', 'mole_flow', 'temp')

    def flatten_states(self):
        pass

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
                    "or the key 'solvent' identifying the passed streams ")

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

    def get_inputs(self, time):
        inputs = {key: get_inputs_new(time, inlet, self.states_in_dict) for
                  key, inlet in self.Inlet.items()}

        return inputs

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

        # # Physical properties
        # rhos = [
        #     self.Liquid_1.getDensity(mole_frac=di_states['x_i'], basis='mole',
        #                              temp=di_states['temp']),
        #     self.Liquid_1.getDensity(mole_frac=di_states['y_i'], basis='mole',
        #                              temp=di_states['temp'])]

        augm_arrays = self.get_augmented_arrays(di_states, inputs)
                                                # bottom_flows)

        holdup_light = np.ones_like(di_states['temp']) * self.fixed_vals['H_R']
        holdup_heavy = np.ones_like(di_states['temp']) * self.fixed_vals['H_E']

        # ---------- Balances
        material = self.material_balances(time,
                                          augm_arrays=augm_arrays,
                                          di_sdot=di_sdot,
                                          holdup_heavy=holdup_heavy,
                                          holdup_light=holdup_light,
                                          **di_states)

        energy = self.energy_balances(time,
                                      augm_arrays=augm_arrays,
                                      holdup_heavy=holdup_heavy,
                                      holdup_light=holdup_light,
                                      di_sdot=di_sdot,
                                      **di_states)

        balances = np.column_stack(material + energy).ravel()

        # print(balances)

        return balances

    def material_balances(self, time, x_i, y_i,
                          holdup_light, holdup_heavy,  # top_flows,
                          u_int, temp,
                          di_sdot, augm_arrays):

        x_augm, y_augm, temp_augm, light_flows, heavy_flows = augm_arrays

        # print(x_augm.sum(axis=1))

        # ---------- Equilibrium
        k_ij = self.k_fun(x_i, y_i, temp)  # TODO: make this stage-wise
        m_ij = k_ij / self.eff

        # ---------- Differential block
        dxij_dt = y_augm[1:] * heavy_flows[1:, np.newaxis] \
            + x_augm[:-1] * light_flows[:-1, np.newaxis] \
            - y_augm[:-1] * heavy_flows[:-1, np.newaxis] \
            - x_augm[1:] * light_flows[1:, np.newaxis]

        div = holdup_light[:, np.newaxis] + holdup_heavy[:, np.newaxis] * m_ij

        dxij_dt *= 1 / div

        # ---------- Algebraic block
        equilibrium_alg = m_ij * (x_augm[1:] - x_augm[:-1] * (1 - self.eff)) \
            - y_i

        # ---------- Modify outputs for stage two on
        if self.num_stages > 1:
            deriv_term = holdup_heavy[0] * m_ij * (1 - self.eff) / div[1:] * \
                dxij_dt[:-1]

            dxij_dt[1:] += deriv_term

            # equilibrium_alg[1:] = m_ij * (x_i[1:] - x_i[:-1] * (1 - self.eff)) \
            #     - y_i[1:]

        if di_sdot is not None:
            dxij_dt = dxij_dt - di_sdot['x_i']

        # nij_alg = x_i * holdup_light[:, np.newaxis] \
        #     + y_i * holdup_heavy[:, np.newaxis] - mol_i

        # equilibrium_alg = m_ij * (x_augm[1:] - x_augm[:-1] * (1 - self.eff)) \
        #     - y_i

        # global_alg = holdup_light + holdup_heavy - mol_i.sum(axis=1)
        # volume_alg = holdup_light/rho_light + holdup_heavy/rho_heavy \
        #     - self.vol * 1000

        out = [dxij_dt, equilibrium_alg]

        # di_flows = {'heavy': {'in': heavy_flows[1], 'out': heavy_flows[0]},
        #             'light': {'in': light_flows[0], 'out': light_flows[1]}}

        # print(di_flows)

        return out

    def energy_balances(self, time, x_i, y_i,
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

        # ---------- Discriminate heavy and light phases
        rhos_streams = {key: obj.getDensity(basis='mole')
                        for key, obj in self.Inlet.items()}

        rhos_holdups = [res.rho_heavy, res.rho_light]

        idx_raff = np.argmin(abs(rhos_holdups - rhos_streams['feed']))

        target_states = {'x_light': 'x_i', 'x_heavy': 'y_i'}
                         # 'mol_light': 'holdup_light',
                         # 'mol_heavy': 'holdup_heavy'}

        if idx_raff == 0:
            target_states['light_phase'] = 'solvent'
            target_states['heavy_phase'] = 'feed'

        elif idx_raff == 1:
            target_states['light_phase'] = 'feed'
            target_states['heavy_phase'] = 'solvent'

        self.target_states = target_states

        # Store fixed values
        self.fixed_vals['H_R'] = res.mol_light
        self.fixed_vals['H_E'] = res.mol_heavy

        # inputs = self.get_inputs(0)

        # # self.fixed_vals['R_i'] = inputs[target_states['light_phase']]['Inlet']['mole_flow']
        # # self.fixed_vals['E_i'] = inputs[target_states['heavy_phase']]['Inlet']['mole_flow']

        # ---------- Create dictionary of initial values
        di_init = {}

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

        rhos_holdups = [
            self.Liquid_1.getDensity(mole_frac=di_init[key], basis='mole',
                                     temp=di_init['temp'])
            for key in keys_frac]

        # self.get_stage_dimensions(di_init,
        #                           [rhos_holdups[0][0], rhos_holdups[1][0]])

        # Energy balance calculations
        h_light = self.Liquid_1.getEnthalpy(mole_frac=di_init['x_i'],
                                            temp=di_init['temp'],
                                            basis='mole')

        h_heavy = self.Liquid_1.getEnthalpy(mole_frac=di_init['y_i'],
                                            temp=di_init['temp'],
                                            basis='mole')

        # u_int = di_init['holdup_light'] * h_light \
        #     + di_init['holdup_heavy'] * h_heavy

        u_int = res.mol_light * h_light + res.mol_heavy * h_heavy

        di_init['u_int'] = u_int

        # ---------- Equilibrium correction for x_i and y_i
        def equilibrium_eqns(fracs, temp, mol_i, holdups):
            var = fracs.reshape(self.num_stages, -1)

            xi = var[:, :self.num_comp]
            yi = var[:, self.num_comp:2 * self.num_comp]

            HR, HE = holdups

            x_eqns = (HR * xi + HE * yi - mol_i) / HR

            k_ij = self.k_fun(xi, yi, temp)

            m_ij = k_ij / self.eff  # TODO: it's not calculated stage-wise (yet)

            y_eqns = np.zeros_like(yi)
            y_eqns[0] = m_ij * xi[0] - yi[0]

            if self.num_stages > 1:
                y_eqns[1:] = m_ij * (xi[1:] - xi[:-1] * (1 - self.eff)) - yi[1:]

            eqns = np.column_stack((x_eqns, y_eqns)).ravel()

            return eqns

        holdups = [res.mol_light, res.mol_heavy]
        mol_i = res.mol_light * res.x_light + res.mol_heavy * res.x_heavy

        args_eq = (di_init['temp'], mol_i, holdups)
        x0 = np.column_stack((di_init['x_i'], di_init['y_i'])).ravel()

        optimopts = {'xtol': 2**(-52)}
        frac_result = root(equilibrium_eqns, x0, args=args_eq)

        frac_noneq = frac_result.x

        frac_noneq = frac_noneq.reshape(self.num_stages, -1)

        x_noneq, y_noneq = np.split(frac_noneq, 2, axis=1)

        di_init['x_i'] = x_noneq
        di_init['y_i'] = y_noneq

        # di_init['x_i'] = di_init['x_i'][:, idx_light]
        # di_init['y_i'] = di_init['y_i'][:, idx_heavy]

        return di_init

    def solve_unit(self, runtime, sundials_opts=None, verbose=True):

        di_init = self.initialize_model()
        di_init = {key: di_init[key] for key in self.name_states}

        init_states = np.column_stack(tuple(di_init.values()))
        init_states = init_states.ravel()

        init_deriv = self.unit_model(0, init_states)

        problem = Implicit_Problem(self.unit_model,
                                   y0=init_states, yd0=init_deriv)

        problem.algvar = self.alg_map

        solver = IDA(problem)
        solver.make_consistent('IDA_YA_YDP_INIT')
        solver.suppress_alg = True

        if not verbose:
            solver.verbosity = 50

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

        self.profiles_runs.append(di)

        self.result = DynamicResult(self.states_di, self.fstates_di, **di)

        # Create time-dependent results for the last stage (outlet)
        di_last = retrieve_pde_result(di, x_name='stage', x=self.num_stages)

        frac_keys = ('x_i', 'y_i')
        frac_last = {key: np.column_stack(
            list(di_last[key].values())) for key in frac_keys}

        inputs = self.get_inputs(time[-1])

        flow_light = inputs[self.target_states['light_phase']]['Inlet']['mole_flow']
        flow_heavy = inputs[self.target_states['heavy_phase']]['Inlet']['mole_flow']

        kws = {
            self.target_states['light_phase']: {
                'mole_flow': flow_light,
                'mole_frac': frac_last['x_i'][-1]},
            self.target_states['heavy_phase']: {
                'mole_flow': flow_heavy,
                'mole_frac': frac_last['y_i'][-1]}
                }

        self.Outlet = {key: LiquidStream(
            self.Liquid_1.path_data, temp=di_last['temp'][-1], **kws[key])
            for key in ('feed', 'solvent')}

        # Complete di_last with appropriate keys
        if self.target_states['light_phase'] == self.default_output:
            di_last['mole_frac'] = frac_last['x_i']
        else:
            di_last['mole_frac'] = frac_last['y_i']

        di_last['mole_flow'] = np.repeat(kws[self.default_output]['mole_flow'],
                                         len(time))

        self.outputs = di_last

        # self.Outlet = LiquidStream(self.Liquid_1.path_data,
        #                            temp=di_last['temp'][-1],
        #                            **kws['feed'])

    def plot_profiles(self, times=None, stages=None, pick_comp=None,
                      **fig_kwargs):

        if pick_comp is None:
            states_plot = ('x_i', 'y_i', 'temp')
            num_species_plot = len(self.name_species)
        else:
            states_plot = (('x_i', pick_comp), ('y_i', pick_comp), 'temp')
            num_species_plot = len(pick_comp)

        ylabels = ('x_i', 'y_i', 'T')

        fig, axis = plot_distrib(self, states_plot, 'stage', times=times,
                                 x_vals=stages, nrows=1, ncols=3,
                                 ylabels=ylabels, **fig_kwargs)

        if times is not None:
            marks = ('s', 'o', '^', 'd', '+')

            for ax in axis:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            for ct, ax in enumerate(axis):
                lines = ax.lines

                for ind, line in enumerate(lines):
                    mark = marks[ind % num_species_plot]

                    line.set_marker(mark)
                    line.set_markerfacecolor('None')

                ax.yaxis.set_minor_locator(AutoMinorLocator(2))

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        elif stages is not None:
            fig.text(0.5, 0, '$t$ (s)', ha='center')

            for ax in axis:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        for ax in axis:
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        fig.tight_layout()

        return fig, axis

