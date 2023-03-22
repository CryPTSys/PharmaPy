import numpy as np
from assimulo.problem import Implicit_Problem
from PharmaPy.Phases import classify_phases
from PharmaPy.Streams import VaporStream
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Commons import (unpack_discretized, retrieve_pde_result,
                              eval_state_events, handle_events)
from PharmaPy.Streams import LiquidStream
from PharmaPy.Results import DynamicResult
from PharmaPy.Plotting import plot_distrib
from PharmaPy.CheckModule import check_modeling_objects

from assimulo.solvers import IDA

import scipy.optimize
import scipy.sparse

# from itertools import cycle
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


class _BaseDistillation:
    def __init__(self, pres, q_feed, LK, HK,
                 perc_LK, perc_HK, reflux=None, num_plates=None,
                 gamma_model='ideal', num_feed=None):

        self.num_plates = num_plates
        self.reflux = reflux
        self.q_feed = q_feed
        self.pres = pres

        self.LK = LK
        self.HK = HK
        self.frac_HK = perc_HK/100
        self.frac_LK = perc_LK/100

        self.gamma_model = gamma_model

        self.num_feed = num_feed  # Num plate from bottom
        self.x_NLK = 1  # Sharp split all NLK recovred in distillate
        self.x_NHK = 0  # Sharp split no NHK in distillate

        self._Inlet = None
        self._Phases = None

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):
        self._Inlet = inlet
        self._Inlet.pres = self.pres
        self.feed_flowrate = inlet.mole_flow
        self.z_feed = inlet.mole_frac

        name_species = self.Inlet.name_species
        self.num_species = len(name_species)
        self.LK_index = name_species.index(self.LK)
        self.HK_index = name_species.index(self.HK)

        self.name_species = name_species

        num_comp = self.num_species
        self.len_in = [1, num_comp]  # , 1]
        self.len_out = [1, num_comp]

        self.nomenclature()

    def nomenclature(self):  # TODO: We might need to make this child-specific
        self.states_di = {
            'temp': {'dim': 1, 'units': 'K'},
            'x_liq': {'dim': len(self.name_species),
                      'index': self.name_species}
            }

        self.states_uo = list(self.states_di.keys())
        self.dim_states = [di['dim'] for di in self.states_di.values()]

        self.fstates_di = {
            'y_vap': {'dim': len(self.name_species),
                      'index': self.name_species}
            }

    def get_inputs(self, time):
        inputs = get_inputs_new(time, self.Inlet, self.states_in_dict)

        return inputs

    def get_alpha(self, pres, x_frac):
        temp_bubble = self.Inlet.getBubblePoint(pres, mole_frac=x_frac)

        k_vals = self.Inlet.getKeqVLE(temp=temp_bubble,
                                      pres=pres, x_liq=x_frac,
                                      gamma_model=self.gamma_model)

        alpha = k_vals/k_vals[self.HK_index]

        return alpha

    def global_material_bce(self, z_feed=None):
        if z_feed is None:
            z_feed = self.z_feed

        HK_index = self.HK_index
        LK_index = self.LK_index

        feed_flow = self.feed_flowrate

        # ---------- Determine Light Key and Heavy Key component numbers
        temp_bubble_feed = self.Inlet.getBubblePoint(pres=self.pres,
                                                     mole_frac=z_feed)

        k_feed = self.Inlet.getKeqVLE(temp=temp_bubble_feed, pres=self.pres)
        volatility_order = np.argsort(k_feed)[::-1]
        self.sorted_by_volatility = [self.name_species[ind]
                                     for ind in volatility_order]

        hk_loc = np.where(volatility_order == HK_index)[0][0]
        lk_loc = np.where(volatility_order == LK_index)[0][0]

        lighter_idx = volatility_order[:lk_loc]
        heavier_idx = volatility_order[hk_loc + 1:]

        if hk_loc != lk_loc + 1:
            print('High key and low key indices are not adjacent', end='\n\n')
            print('Volatility order at T_bubble = %.1f K (%.0f Pa, high to low): ' % (temp_bubble_feed, self.pres) +
                  '-'.join(self.sorted_by_volatility))

        # ---------- Calculate Distillate and Bottom flow rates
        bot_flow = feed_flow * (z_feed[HK_index] * ((1 - self.frac_HK)) +
                                z_feed[LK_index] * ((1 - self.frac_LK)) +
                                z_feed[lighter_idx].sum() * (1 - self.x_NLK) +
                                z_feed[heavier_idx].sum() * (1 - self.x_NHK))

        dist_flow = feed_flow - bot_flow

        if bot_flow < 0 or dist_flow < 0:
            print('negative flow rates, given value not feasible')

        # ---------- Estimate component fractions
        x_dist = np.zeros_like(z_feed)
        x_bot = np.zeros_like(z_feed)

        x_bot[lighter_idx] = feed_flow * z_feed[lighter_idx] * \
            (1 - self.x_NLK) / bot_flow
        x_bot[LK_index] = feed_flow * z_feed[LK_index] * \
            (1 - self.frac_LK) / bot_flow
        x_bot[HK_index] = feed_flow * z_feed[HK_index] * \
            (1 - self.frac_HK) / bot_flow
        x_bot[heavier_idx] = feed_flow * z_feed[heavier_idx] * \
            (1 - self.x_NHK) / bot_flow

        x_dist = (feed_flow*z_feed - bot_flow*x_bot) / dist_flow

        return x_dist, x_bot, dist_flow, bot_flow

    def calc_num_min(self, x_dist, x_bot):
        LK_index = self.LK_index

        # Fenske equation
        alpha_top = self.get_alpha(self.pres, x_dist)
        alpha_bottom = self.get_alpha(self.pres, x_bot)
        alpha_fenske = (alpha_top[LK_index] * alpha_bottom[LK_index])**0.5

        num_min = np.log(
            self.frac_LK / (1 - self.frac_LK) * (1 - self.frac_HK) / self.frac_HK) / \
            np.log(alpha_fenske)

        return num_min

    def molokanov_equation(self, reflux, min_reflux, num_min):
        x_val = (reflux - min_reflux) / (reflux + 1)

        y_val = 1 - np.exp((1 + 54.4*x_val) / (11 + 117.2*x_val) *
                           (x_val - 1)/np.sqrt(x_val))

        num_stages = (y_val + num_min) / (1 - y_val)

        return num_stages

    def kirkbride_correlation(self, material_bce, num_plates, z_feed=None):
        if z_feed is None:
            z_feed = self.z_feed

        z_lk = z_feed[self.LK_index]
        z_hk = z_feed[self.HK_index]

        x_dist = material_bce['x_dist']
        x_bot = material_bce['x_bottom']

        bot_flow = material_bce['bottom_flow']
        dist_flow = material_bce['dist_flow']

        num_ratio = ((z_hk / z_lk) *
                     (x_bot[self.LK_index] / x_dist[self.HK_index])**2 *
                     (bot_flow / dist_flow))**0.206

        num_strip = num_plates / (num_ratio + 1)
        num_rect = num_plates - num_strip

        return np.round(num_rect), np.round(num_strip)

    def calc_min_reflux(self, x_dist, x_bot, dist_flowrate, bot_flowrate,
                        z_feed=None):
        if z_feed is None:
            z_feed = self.z_feed

        LK_index = self.LK_index
        HK_index = self.HK_index

        alpha = self.get_alpha(self.pres, z_feed)

        def f(phi):
            fun = (sum(alpha * z_feed /
                       (alpha - phi + np.finfo(float).eps)))**2

            return fun

        bounds = ((alpha[HK_index], alpha[LK_index]), )

        phi = scipy.optimize.minimize(
            f, (alpha[LK_index] + alpha[HK_index])/2, bounds=bounds, tol=1e-10)
        phi = phi.x

        # Second underwood equation
        V_min = sum(alpha * dist_flowrate * x_dist / (alpha - phi))
        L_min = V_min - dist_flowrate

        min_reflux = L_min/dist_flowrate

        return min_reflux

    def calculate_heuristics(self, time=None):

        if time is None:
            z_feed = self.z_feed

        else:
            inputs = self.get_inputs(time)
            z_feed = inputs['Inlet']['mole_frac']

        x_dist, x_bot, dist_flowrate, bot_flowrate = self.global_material_bce(
            z_feed)

        min_reflux = self.calc_min_reflux(x_dist, x_bot,
                                          dist_flowrate, bot_flowrate,
                                          z_feed)

        num_min = self.calc_num_min(x_dist, x_bot)

        mat_bce = {'x_dist': x_dist, 'x_bottom': x_bot,
                   'dist_flow': dist_flowrate, 'bottom_flow': bot_flowrate}

        # Actual reflux
        if self.reflux is None or self.reflux == 0:
            print('Using reflux = 1.5 * min_reflux')
            reflux = 1.5 * min_reflux  # Heuristic
        elif self.reflux < 0:
            reflux = -self.reflux * min_reflux
        elif self.reflux > 0 and self.reflux < min_reflux:
            print(
                'Specified reflux less than min_reflux, calculation proceeds '
                'with 1.5 * min_reflux')

            reflux = 1.5 * self.min_reflux

        else:
            reflux = self.reflux

        # Actual number of stages
        if self.num_plates == 0 or self.num_plates is None:
            print('Using Gilliland correlation for number of stages')
            num_plates = self.molokanov_equation(reflux, min_reflux, num_min)
            num_plates = np.ceil(num_plates)
        elif self.num_plates < 0:
            num_plates = np.ceil(num_min * abs(self.num_plates))
        elif self.num_plates > 0:
            if self.num_plates < np.ceil(num_min):
                print('Warning: Specified number of plates (%i) is lower than '
                      'the calculated minimum number of plates (%i)'
                      % (self.num_plates, int(np.ceil(num_min)))
                      )
            num_plates = self.num_plates

        # Feed stage
        if self.num_feed is None:
            num_rect, num_strip = self.kirkbride_correlation(
                mat_bce, num_plates, z_feed)

            if num_rect > num_strip:
                num_feed = num_rect
                num_rect -= 1
            else:
                num_feed = num_rect + 1
                num_strip -= 1

        elif self.num_feed > num_plates:
            raise ValueError(
                'Feed number is larger than total number of plates')
        else:
            num_feed = self.num_feed

        out = {'material_balances': mat_bce,
               'min_reflux': min_reflux, 'num_min': num_min,
               'reflux': reflux, 'num_plates': num_plates,
               'num_feed': num_feed}

        return out


class DistillationColumn(_BaseDistillation):
    def __init__(self, pres, q_feed, LK, HK,
                 perc_LK, perc_HK, reflux=None, num_plates=None,
                 gamma_model='ideal', num_feed=None):

        """
        Create an object to solve a steady-state distillation column

        Parameters
        ----------
        pres : float
            column pressure [Pa].
        q_feed : float
            fraction of liquid in the feed stream in molar basis.
        LK : str
            name of the light key component.
        HK : str
            name of the heavy key component.
        perc_LK : float
            percentage of the light key component in the feed that appears in
            the top product.
        perc_HK : float
            percentage of the heavy key component in the feed that appears in
            the top product.
        reflux : float, optional
            reflux ratio (L/D), with L the internal liquid flow and D the
            distillate flow. If None, reflux = Rmin * 1.5, where Rmin is
            calculated using the Underwood equation. If negative,
            reflux = Rmin * abs(reflux). The default is None.
        num_plates : int, optional
            number of equilibrium stages. If None, num_plates will be estimated
            using the Fenske equation for Nmin and the Gilliland method for N.
            If negative, num_plates = Nmin * abs(num_plates).
            The default is None.
        gamma_model : str, optional
            name of the thermodynamic model used for activity coefficient
            calculation. It can be 'UNIQUAC', 'UNIFAC' or 'ideal'.
            The default is 'ideal'.
        num_feed : int, optional
            feed tray (trays are numbered from top to bottom).
            The default is None.

        Returns
        -------
        PharmaPy DistillationColumn object.

        """

        super().__init__(pres, q_feed, LK, HK, perc_LK, perc_HK, reflux,
                         num_plates, gamma_model, num_feed)

    def VLE(self, y_oneplate=None, temp=None, need_x_vap=True):
        # VLE uses vapor stream, need vapor stream object temporarily.
        temporary_vapor = VaporStream(path_thermo=self._Inlet.path_data,
                                      pres=self.pres, mole_flow=self.feed_flowrate, mole_frac=y_oneplate)
        res = temporary_vapor.getDewPoint(pres=self.pres, mass_frac=None,
                                          mole_frac=y_oneplate, thermo_method=self.gamma_model, x_liq=need_x_vap)
        # Program needs VLE function to return output in x,Temp format
        return res[::-1]


    def calc_plates(self, x_dist, x_bottom, dist_flow, bottom_flow, reflux, num_plates):
        LK_index = self.LK_index
        HK_index = self.HK_index

        # Calculate Vapour and Liquid flows in column
        # Rectifying section
        Ln = reflux*dist_flow
        Vn = Ln + dist_flow

        # Stripping section
        Lm = Ln + self.feed_flowrate + self.feed_flowrate*(self.q_feed-1)
        Vm = Lm - bottom_flow

        if num_plates is None:
            # Calculate number of plates
            # Composition list
            x = []
            y = []
            T = []
            counter1 = 1
            counter2 = 1
            # more likely to have non LKs and no non HKs
            # start counting from top of column
            # First plate
            y.append(x_dist)
            x_new, T_new = self.VLE(y[0])
            x.append(x_new)
            T.append(T_new)

            y_bot_op = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bottom)
            y_top_op = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)

            # Rectifying section
            while (y_top_op[LK_index]/y_top_op[HK_index] < y_bot_op[LK_index]/y_bot_op[HK_index]):
                y.append((np.array(x_new) * Ln/Vn + (1 - Ln/Vn) * x_dist))
                x_new, T_new = self.VLE(y[-1])
                x.append(x_new)
                T.append(T_new)

                if counter2 > 100:
                    break
                counter2 += 1
                y_bot_op = (np.array(x_new) * Lm/Vm - (Lm/Vm - 1) * x_bottom)
                y_top_op = (np.array(x_new) * Ln/Vn + (1 - Ln/Vn) * x_dist)

            # Feed plate
            num_feed = counter2

            # Stripping section
            # When reflux is specified and num_plates is not calculated x returned contains one extra evaluation, not true for case when num_plates is specified. This is why fudge factors are added
            while (np.array(x[-1][HK_index]) < 0.98*x_bottom[HK_index] or np.array(x[-1][LK_index]) > 1.2*x_bottom[LK_index]):
                y.append((np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bottom))
                x_new, T_new = self.VLE(y[-1])
                x.append(x_new)
                T.append(T_new)
                if counter1 > 100:
                    break
                counter1 += 1

            num_plates = len(y) - 1  # Remove distillate stream, reboiler

        else:  # Num plates specified

            # Calculate compositions
            x = np.zeros((num_plates + 1, self.num_species))
            y = np.zeros((num_plates + 1, self.num_species))
            T = np.zeros(num_plates + 1)

            y[0] = x_dist
            x_new, T_new = self.VLE(y[0])
            x[0] = x_new
            T[0] = T_new

            if self.num_feed is None:  # Feed plate not specified
                y_bot_op = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bottom)
                y_top_op = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)
                flag_rect_section_present = 0

                for i in range(1, num_plates+1):
                    # Rectifying section
                    if (y_top_op[LK_index]/y_top_op[HK_index] < y_bot_op[LK_index]/y_bot_op[HK_index]):

                        y[i] = (np.array(x_new)*Ln/Vn + (1 - Ln/Vn)*x_dist)
                        x_new, T_new = self.VLE(y[i])
                        x[i] = x_new
                        T[i] = T_new

                        y_bot_op = (np.array(x_new)*Lm/Vm - (Lm/Vm - 1)*x_bottom)
                        y_top_op = (np.array(x_new)*Ln/Vn + (1 - Ln/Vn)*x_dist)
                        num_feed = i  # To get feed plate, only last value used, cant write outside if cause elif structure will be violated
                        flag_rect_section_present = 1

                    # Stripping section
                    elif (np.array(x[-1][HK_index]) < 0.98*x_bottom[HK_index] or np.array(x[-1][LK_index]) > 1.2*x_bottom[LK_index]):
                        y[i] = (np.array(x_new)*Lm/Vm - (Lm/Vm - 1)*x_bottom)
                        x_new, T_new = self.VLE(y[i])
                        x[i] = x_new
                        T[i] = T_new
                        if not(flag_rect_section_present):
                            num_feed = num_plates
            else:  # Feed plate specified
                num_feed = self.num_feed
                for i in range(1, num_plates + 1):
                    # Rectifying section
                    if i < self.num_feed:
                        y[i] = (np.array(x_new)*Ln/Vn + (1 - Ln/Vn)*x_dist)
                        x_new, T_new = self.VLE(y[i])
                        x[i] = x_new
                        T[i] = T_new
                        num_feed = self.num_feed
                    # Stripping section
                    else:
                        y[i] = (np.array(x_new)*Lm/Vm - (Lm/Vm - 1)*x_bottom)
                        x_new, T_new = self.VLE(y[i])
                        x[i] = x_new
                        T[i] = T_new

        self.num_feed = num_feed

        self.retrieve_results(num_plates, x, y, T, bottom_flow, dist_flow,
                              reflux, num_feed, x_dist, x_bottom,
                              self.min_reflux, self.num_min)
        return num_plates

    def solve_unit(self, runtime=None, t0=0, solve_ss=True):
        result = self.calculate_heuristics()

        self.min_reflux = result['min_reflux']
        self.num_min = result['num_min']
        self.reflux = result['reflux']
        self.num_plates = result['num_plates']

        # out = {'material_balances': mat_bce,
        #        'min_reflux': min_reflux, 'num_min': num_min,
        #        'reflux': reflux, 'num_plates': num_plates}

        if solve_ss:
            self.calc_plates(**result['material_balances'],
                             reflux=result['reflux'],
                             num_plates=result['num_plates'])
            self.retrieve_results()

        return result

    def retrieve_results(self, num_plates, x, y, T,
                         bot_flowrate, dist_flowrate, reflux, num_feed,
                         x_dist, x_bot, min_reflux, N_min):

        if not(isinstance(x, np.ndarray)):
            x = np.array(x)
            y = np.array(y)
            T = np.array(T)

        dist_result = {'num_plates': num_plates, 'x': x.T, 'y': y.T, 'T': T,
                       'bot_flowrate': bot_flowrate, 'dist_flowrate': dist_flowrate, 'reflux': reflux,
                       'num_feed': num_feed, 'x_dist': x_dist, 'x_bot': x_bot, 'min_reflux': min_reflux, 'N_min': N_min
                       }

        self.result = DynamicResult(self.states_di, **dist_result)

        path = self.Inlet.path_data
        self.OutletDistillate = LiquidStream(path, temp=dist_result['T'][0],
                                             mole_conc=dist_result['x_dist'],
                                             mole_flow=dist_result['dist_flowrate'])
        self.OutletBottom = LiquidStream(path, temp=dist_result['T'][-1],
                                         mole_conc=dist_result['x_bot'],
                                         mole_flow=dist_result['bot_flowrate'])
        self.Outlet = self.OutletBottom


class DynamicDistillation(_BaseDistillation):
    def __init__(self, pres, q_feed, LK, HK,
                 perc_LK, perc_HK, reflux=None, num_plates=None,
                 gamma_model='ideal', num_feed=None, state_events=None):
        """ Create an object to solve a dynamic distillation column

        Parameters
        ----------
        pres : float
            column pressure [Pa].
        q_feed : float
            fraction of liquid in the feed stream in molar basis.
        LK : str
            name of the light key component.
        HK : str
            name of the heavy key component.
        x_LK : float
            desired mole fraction of the light key in the top product.
        x_HK : float
            desired mole fraction of the heavy key in the top product.
        reflux : float, optional
            reflux ratio (L/D), begin L the internal liquid flow and D the
            distillate flow. The default is None.
        num_plates : int, optional
            number of equilibrium stages. If not provided, it will be estimated
            using the YYY method. The default is None.
        gamma_model : str, optional
            name of the thermodynamic model used for activity coefficient
            calculation. It can be 'UNIQUAC', 'UNIFAC' or 'ideal'.
            The default is 'ideal'.
        num_feed : int, optional
            feed tray (trays are numbered from top to bottom).
            The default is None.

        Returns
        -------
        PharmaPy DistillationColumn object.

        """

        super().__init__(pres, q_feed, LK, HK, perc_LK, perc_HK, reflux,
                         num_plates, gamma_model, num_feed)

        self.oper_mode = 'Continuous'
        self.outputs = None
        self.is_continuous = True
        self.elapsed_time = 0

        if state_events is None:
            state_events = []
        elif isinstance(state_events, dict):
            state_events = [state_events]

        self.state_event_list = state_events

    def flatten_states(self):
        pass

    def nomenclature(self):
        super().nomenclature()

        self.name_states = list(self.states_di.keys())
        self.names_states_out = ['temp', 'mole_frac', 'mole_flow']
        self.names_states_in = self.names_states_out

        num_comp = len(self.name_species)
        len_in = [1, num_comp, 1]

        states_in_dict = dict(zip(self.names_states_in, len_in))
        self.states_in_dict = {'Inlet': states_in_dict}

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        if not isinstance(phases, (list, tuple)):
            phases = [phases]

        self._Phases = phases

        classify_phases(self)

        self.holdup = self.Liquid_1.moles

        name_species = self.Liquid_1.name_species
        self.num_species = len(name_species)
        self.LK_index = name_species.index(self.LK)
        self.HK_index = name_species.index(self.HK)

        self.name_species = name_species

        self.nomenclature()

    def column_startup(self, time_heuristics):
        result = self.calculate_heuristics(time_heuristics)

        global_mbce = result['material_balances']

        self.num_plates = int(result['num_plates'])
        self.num_feed = int(result['num_feed'])

        self.bot_flowrate = global_mbce['bottom_flow']
        self.dist_flowrate = global_mbce['dist_flow']
        self.reflux = result['reflux']

        self.x_dist = global_mbce['x_dist']
        self.x_bot = global_mbce['x_bottom']

        self.min_reflux = result['min_reflux']
        self.num_min = result['num_min']

        self.heuristics = result

    def _eval_state_events(self, time, states, sdot, sw):
        events = eval_state_events(
            time, states, sw, self.dim_states,
            self.states_uo, self.state_event_list, sdot=sdot,
            discretized_model=True)

        return events

    def unit_model(self, time, states, d_states, sw=None):
        di_states = unpack_discretized(states, self.len_out, self.name_states)

        material = self.material_balances(time, **di_states)

        di_d_states = unpack_discretized(d_states, self.len_out,
                                         self.name_states)
        # N_plates(N_components), only for compositions
        material[:, 1:] = material[:, 1:] - di_d_states['x_liq']
        balances = material.ravel()

        # print(balances)
        return balances

    def material_balances(self, time, temp, x_liq):
        x = x_liq
        inputs = self.get_inputs(time)['Inlet']
        z_feed = inputs['mole_frac']
        feed_flow = inputs['mole_flow']

        # GET STARTUP CONDITIONS
        (bot_flowrate, dist_flowrate,
         reflux, num_feed, M_const) = (self.bot_flowrate, self.dist_flowrate,
                                       self.reflux, self.num_feed, self.holdup)

        # CALCULATE COLUMN FLOWS
        # Rectifying section
        Ln = reflux * dist_flowrate
        Vn = Ln + dist_flowrate

        # Stripping section
        Lm = Ln + self.feed_flowrate * self.q_feed
        Vm = Lm - bot_flowrate

        dx_dt = np.zeros_like(x)

        k_vals = self._Inlet.getKeqVLE(pres=self.pres, temp=temp, x_liq=x)

        residuals_temp = (x * (k_vals - 1)).sum(axis=1)
        y = k_vals * x

        # Reflux tank
        dx_dt[0] = Vn/M_const * (y[1] - x[0])

        # Rectifying section
        dx_dt[1:num_feed] = 1/M_const * (
            Vn * y[2:num_feed + 1] + Ln * x[:num_feed - 1] -
            Vn * y[1:num_feed] - Ln * x[1:num_feed])

        # Feed plate
        dx_dt[num_feed] = 1/M_const * (
            Vm * y[num_feed + 1] + Ln * x[num_feed - 1] + feed_flow * z_feed -
            Vn * y[num_feed] - Lm * x[num_feed])

        # Stripping section
        dx_dt[num_feed + 1:-1] = 1/M_const * (
            Vm * y[num_feed + 2:] + Lm * x[num_feed:-2] -
            Vm * y[num_feed + 1:-1] - Lm*x[num_feed + 1:-1])

        # Reboiler, y_in for reboiler is the same as x_out
        dx_dt[-1] = 1/M_const * (Lm*x[-2] - Vm*y[-1] - bot_flowrate*x[-1])
        mat_bal = np.column_stack((residuals_temp, dx_dt))

        return mat_bal

    def energy_balances(self, time, temp, mole_frac):
        pass
        return

    def solve_unit(self, runtime=None, time_grid=None,
                   sundials_opts=None, verbose=True, any_event=True):

        check_modeling_objects(self)

        if runtime is not None and time_grid is not None:
            raise RuntimeError("Both 'runtime' and 'time_grid' were provided. "
                               "Please provide only one of them")
        elif runtime is not None:
            final_time = runtime + self.elapsed_time

        elif time_grid is not None:
            final_time = time_grid[-1] + self.elapsed_time

        # Use inputs coming from the upstream UO in the future for heuristics,
        # i.e. as close to steady-state as possible
        self.column_startup(final_time)

        self.len_states = len(self.name_species) + 1

        x_init = self.Liquid_1.mole_frac.copy()
        temp_init = self.Liquid_1.getBubblePoint(pres=self.pres,
                                                 mole_frac=x_init)

        init_states = np.tile(np.hstack((temp_init, x_init)),
                              (self.num_plates + 1, 1))

        init_derivative = self.material_balances(time=self.elapsed_time,
                                                 x_liq=init_states[:, 1:],
                                                 temp=init_states[:, 0])

        if len(self.state_event_list) > 0:
            def new_handle(solver, info):
                return handle_events(solver, info, self.state_event_list,
                                     any_event=any_event)

            switches = [True] * len(self.state_event_list)
            problem = Implicit_Problem(
                self.unit_model, init_states.ravel(), init_derivative.ravel(),
                t0=self.elapsed_time, sw0=switches)

            problem.state_events = self._eval_state_events
            problem.handle_event = new_handle

        else:
            problem = Implicit_Problem(
                self.unit_model, init_states.ravel(), init_derivative.ravel(),
                t0=self.elapsed_time)

        solver = IDA(problem)
        solver.make_consistent('IDA_YA_YDP_INIT')
        # solver.suppress_alg = True

        alg_map = np.zeros_like(init_states)
        alg_map[:, 0] = 1

        solver.algvar = alg_map.ravel()

        if not verbose:
            solver.verbosity = 50

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        time, states, d_states = solver.simulate(final_time,
                                                 ncp_list=time_grid)

        self.retrieve_results(time, states)
        return time, states, d_states

    def retrieve_results(self, time, states):
        time = np.asarray(time)
        self.elapsed_time = time[-1]

        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        dp = unpack_discretized(states, self.len_out, self.name_states,
                                indexes=indexes, inputs=None)

        dp['time'] = time
        dp['plate'] = np.arange(self.num_plates + 1)

        # y_vap
        y_vap = {key: [] for key in dp['x_liq']}
        for ind, row in enumerate(dp['temp']):
            x_liquid = np.column_stack(
                [val[ind] for val in dp['x_liq'].values()])

            k_vals = self.Liquid_1.getKeqVLE(pres=self.pres, temp=row,
                                             x_liq=x_liquid)
            y_vals = k_vals * x_liquid

            for idx, key in enumerate(y_vap):
                y_vap[key].append(y_vals[:, idx])

        y_vap = {key: np.vstack(val) for key, val in y_vap.items()}
        dp['y_vap'] = y_vap

        self.result = DynamicResult(di_states=self.states_di,
                                    di_fstates=self.fstates_di, **dp)

        outputs = retrieve_pde_result(self.result, x_name='plate',
                                      x=self.num_plates)

        outputs['x_liq'] = np.column_stack(list(outputs['x_liq'].values()))
        outputs['mole_frac'] = outputs.pop('x_liq')

        outputs['mole_flow'] = np.ones_like(outputs['temp']) * self.bot_flowrate

        self.outputs = outputs
        # [component_index, time, plate]
        x_comp = np.array(list(dp['x_liq'].values()))

        # Outlet stream
        path = self.Inlet.path_data
        self.OutletBottom = LiquidStream(
            path, temp=dp['temp'][-1][-1],  # [time, plate]
            mole_frac=x_comp.T[-1][-1],  # [plate, time, component_index]
            mole_flow=self.bot_flowrate)

        self.OutletDistillate = LiquidStream(
            path, temp=dp['temp'][-1][0],  # [time,plate]
            mole_frac=x_comp.T[0][-1],
            mole_flow=self.dist_flowrate)

        self.Outlet = self.OutletBottom

    def plot_profiles(self, times=None, plates=None, pick_comp=None, **fig_kw):
        states = []
        ylab = ['x_liq', 'T']

        if pick_comp is None:
            states.append('x_liq')
        else:
            states.append(['x_liq', pick_comp])

        states.append('temp')

        if pick_comp is None:
            num_species_plot = self.num_species
        else:
            num_species_plot = len(pick_comp)

        fig, ax = plot_distrib(self, states, 'plate', times=times,
                               x_vals=plates, ylabels=ylab, ncols=2, **fig_kw)

        if times is not None:
            marks = ('s', 'o', '^', '*', '+', '<', '1', 'p', 'd', 'X')

            for ct, axis in enumerate(ax):
                lines = axis.lines

                for ind, line in enumerate(lines):
                    mark = marks[ind % num_species_plot]

                    line.set_marker(mark)
                    line.set_markerfacecolor('None')

                axis.yaxis.set_minor_locator(AutoMinorLocator(2))

                axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        elif plates is not None:
            pass  # TODO

        fig.tight_layout()

        return fig, ax
