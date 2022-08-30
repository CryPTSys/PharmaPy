import numpy as np
from assimulo.problem import Implicit_Problem
from PharmaPy.Phases import classify_phases
from PharmaPy.Streams import VaporStream
from PharmaPy.Connections import get_inputs_new
from PharmaPy.Commons import (trapezoidal_rule, unpack_discretized,
                              retrieve_pde_result)
from PharmaPy.Streams import LiquidStream
from PharmaPy.Results import DynamicResult
from assimulo.solvers import IDA
#Import connectivity

from PharmaPy.Commons import reorder_pde_outputs

import scipy.optimize
import scipy.sparse

class DistillationColumn:
    def __init__(self, name_species, col_P, q_feed, LK, HK, 
                 per_LK, per_HK, reflux=None, num_plates=None, 
                 holdup=None, gamma_model='ideal', N_feed=None):

        self.num_plates = num_plates
        self.name_species = name_species
        self.reflux = reflux
        self.q_feed = q_feed
        self.col_P = col_P
        self.LK = LK
        self.HK = HK
        self.per_HK = per_HK
        self.per_LK = per_LK
        self.num_species = len(name_species)
        self.M_const = holdup
        self.gamma_model= gamma_model
        self.N_feed=N_feed #Num plate from bottom
        self.per_NLK=100 #Sharp split all NLK recovred in distillate
        self.per_NHK = 0 #Sharp split no NHK in distillate
        
        self.nomenclature()
        self._Inlet = None
        self._Phases = None
        return
    
    def nomenclature(self):
        self.name_states = []
        self.names_states_out = []
        self.names_states_in = self.names_states_out
        self.states_di = {
            'num_plates':{'dim':1}, 'x':{'dim':len(self.name_species), 'index': self.name_species}, 'y':{'dim':len(self.name_species),  'index': self.name_species}, 'T':{'dim':1, 'units': 'K'}, 
                       'bot_flowrate':{'dim':1, 'units': 'mole/sec'}, 'dist_flowrate':{'dim':1, 'units': 'mole/sec'}, 'reflux':{'dim':1}, 
                       'N_feed':{'dim':1}, 'x_dist':{'dim':len(self.name_species)}, 'x_bot':{'dim':len(self.name_species)}, 'min_reflux':{'dim':1}, 'N_min':{'dim':1}
            }

    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):
        self._Inlet = inlet
        self._Inlet.pres = self.col_P
        self.feed_flowrate = inlet.mole_flow
        self.z_feed = inlet.mole_frac
        
        num_comp = len(self.Inlet.name_species)
        len_in = [1, num_comp]
        
        if self.names_states_in:
            states_in_dict = dict(zip(self.names_states_in, len_in))
        else:
            states_in_dict = []
        self.states_in_dict = {'Inlet': states_in_dict}
        
    def get_inputs(self, time):
        inputs = get_inputs_new(time, self.Inlet, self.states_in_dict)

        return inputs

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        classify_phases(self)
        #self.nomenclature()
        
    def estimate_comp (self):
        ### Determine Light Key and Heavy Key component numbers
        # Assume z_feed and name species are in the order- lightest to heaviest
        name_species = self.name_species
        feed_flowrate = self.feed_flowrate
        z_feed = self.z_feed
        LK = self.LK
        HK = self.HK
        LK_index = name_species.index(LK)
        HK_index = name_species.index(HK)
        self.LK_index = LK_index
        self.HK_index = HK_index
        if HK_index != LK_index +1:
            print ('High key and low key indices are not adjacent')
        
        ### Calculate Distillate and Bottom flow rates
        bot_flowrate = (feed_flowrate*z_feed[HK_index]*(1-self.per_HK/100) 
                        + feed_flowrate*z_feed[LK_index]*(1-self.per_LK/100)
                        + sum(feed_flowrate*z_feed[:LK_index])*(1-self.per_NLK/100)
                        + sum(feed_flowrate*z_feed[HK_index+1:])*(1-self.per_NHK/100))
        dist_flowrate = feed_flowrate - bot_flowrate
        
        if bot_flowrate <0 or dist_flowrate<0:
            print('negative flow rates, given value not feasible')
        
        ### Estimate component fractions
        x_dist = np.zeros_like(z_feed)
        x_bot = np.zeros_like(z_feed)
        
        x_bot[:LK_index] = (sum(feed_flowrate*z_feed[:LK_index])
                            *(1-self.per_NLK/100)/bot_flowrate)
        x_bot[LK_index]  = (feed_flowrate*z_feed[LK_index]
                            *(1-self.per_LK/100)/bot_flowrate)
        x_bot[HK_index]  = (feed_flowrate*z_feed[HK_index]
                            *(1-self.per_HK/100)/bot_flowrate)
        x_bot[HK_index+1:] = (sum(feed_flowrate*z_feed[HK_index+1:])
                            *(1-self.per_NHK/100)/bot_flowrate)
        
        x_dist = (feed_flowrate*z_feed - bot_flowrate*x_bot)/dist_flowrate
        
        #Fenske equation
        k_vals_bot = self._Inlet.getKeqVLE(pres = self.col_P, x_liq = x_bot)
        k_vals_dist = self._Inlet.getKeqVLE(pres = self.col_P, x_liq = x_dist)
        alpha_fenske = (k_vals_dist[LK_index]/k_vals_dist[HK_index]*
                        k_vals_bot[LK_index]/k_vals_bot[HK_index])**0.5
        N_min = (np.log(self.per_LK/100/(1-self.per_LK/100)/
                        ((self.per_HK/100)/(1-self.per_HK/100)))
                             /np.log(alpha_fenske))
        self.N_min = N_min
        return x_dist, x_bot, dist_flowrate, bot_flowrate
    
    def get_k_vals(self, x_oneplate=None, temp = None):
        if x_oneplate is None:
            x_oneplate = self.z_feed
        k_vals = self._Inlet.getKeqVLE(pres = self.col_P, temp=temp,
                                       x_liq = x_oneplate)
        return k_vals
        
    def VLE(self, y_oneplate=None, temp = None, need_x_vap=True):
        # VLE uses vapor stream, need vapor stream object temporarily.
        temporary_vapor = VaporStream(path_thermo=self._Inlet.path_data, pres=self.col_P, mole_flow=self.feed_flowrate, mole_frac=y_oneplate)
        res = temporary_vapor.getDewPoint(pres=self.col_P, mass_frac=None, 
                    mole_frac=y_oneplate, thermo_method=self.gamma_model, x_liq=need_x_vap)
        return res[::-1] #Program needs VLE function to return output in x,Temp format
    
    def calc_reflux(self, x_dist = None, x_bot = None, 
                    reflux = None, num_plates = None, pres=None, temp=None):
        ### Calculate operating lines
        #Rectifying section
        if (x_dist is None or x_bot is None):
            x_dist, x_bot, dist_flowrate, bot_flowrate = self.estimate_comp()                                            
        LK_index = self.LK_index
        HK_index = self.HK_index
        k_vals = self.get_k_vals(x_oneplate = self.z_feed, temp=temp) 
        
        alpha = k_vals/k_vals[HK_index]
        ### Estimate Reflux ratio
        # First Underwood equation
        f = lambda phi: (sum(alpha * self.z_feed/(alpha - phi + np.finfo(float).eps)))**2 #(1-q is 0), Feed flow rate cancelled from both sides, 10^-10 is to avoid division by 0
        bounds = ((alpha[HK_index], alpha[LK_index]),)
        phi = scipy.optimize.minimize(f, (alpha[LK_index] + alpha[HK_index])/2, bounds = bounds, tol = 10**-10)
        phi = phi.x
        #Second underwood equation
        V_min = sum(alpha*dist_flowrate*x_dist/(alpha-phi))
        L_min = V_min - dist_flowrate
        min_reflux = L_min/dist_flowrate
        self.min_reflux = min_reflux
        
        if reflux is None:
            reflux = 1.5*min_reflux #Heuristic
        
        if reflux <0:
            reflux = -1*reflux*self.min_reflux 
        
        if num_plates is not None:
            def bot_comp_err (reflux, num_plates, x_dist, x_bot, 
                              dist_flowrate, bot_flowrate):
                
                Ln  = reflux*dist_flowrate
                Vn  = Ln  + dist_flowrate
                
                #Stripping section
                Lm  = Ln + self.feed_flowrate + self.feed_flowrate*(self.q_feed-1) 
                Vm  = Lm  - bot_flowrate
                
                #Calculate compositions
                x = np.zeros((num_plates+1, self.num_species))
                y = np.zeros((num_plates+1, self.num_species))
                T = np.zeros(num_plates+1)
                y[0] = x_dist
                x_new,T_new = self.VLE(y[0])
                x[0] = x_new
                T[0] = T_new
                
                if self.N_feed is None: #Feed plate not specified
                    y_bot_op = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bot)
                    y_top_op = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)
                    
                    for i in range (1, num_plates+1):
                        #Rectifying section
                        if (y_top_op[LK_index]/y_top_op[HK_index] < y_bot_op[LK_index]/y_bot_op[HK_index]):
                            
                            y[i] = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)
                            x_new,T_new = self.VLE(y[i])
                            x[i] = x_new
                            T[i] = T_new
                            
                            y_bot_op = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bot)
                            y_top_op = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)
                                
                        #Stripping section
                        else:
                            y[i] = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bot)
                            x_new,T_new = self.VLE(y[i])
                            x[i] = x_new
                            T[i] = T_new
                
                else: #Feed plate specified
                    for i in range (1, num_plates+1):
                        #Rectifying section
                        if i < num_plates+1 - self.N_feed:
                            y[i] = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)
                            x_new,T_new = self.VLE(y[i])
                            x[i] = x_new
                            T[i] = T_new
                                
                        #Stripping section
                        else:
                            y[i] = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bot)
                            x_new,T_new = self.VLE(y[i])
                            x[i] = x_new
                            T[i] = T_new
                            
                error = np.linalg.norm(x_bot - x[-1])/np.linalg.norm(x_bot)*100 + 0.01 * reflux**2 #pentalty for very high reflux values
                return error
            
            reflux = scipy.optimize.minimize(bot_comp_err, x0 = 1.5* self.min_reflux, 
                                                 args=(num_plates, x_dist, x_bot, dist_flowrate, bot_flowrate), 
                                                 method = 'Nelder-Mead')
            reflux = reflux.x
        return reflux, x_dist, x_bot, dist_flowrate, bot_flowrate
    
    def calc_plates(self, reflux = None, num_plates = None):
        reflux, x_dist, x_bot, dist_flowrate, bot_flowrate = self.calc_reflux(reflux = reflux, num_plates  = num_plates)
        LK_index = self.LK_index
        HK_index = self.HK_index
        
        ### Calculate Vapour and Liquid flows in column
        #Rectifying section
        Ln  = reflux*dist_flowrate
        Vn  = Ln  + dist_flowrate
        
        #Stripping section
        Lm  = Ln + self.feed_flowrate + self.feed_flowrate*(self.q_feed-1) 
        Vm  = Lm  - bot_flowrate
        
        
        if num_plates is None:
            ### Calculate number of plates
            #Composition list
            x = []
            y = []
            T = []
            counter1= 1
            counter2 =1
            #more likely to have non LKs and no non HKs
            #start counting from top of column
            #First plate
            y.append(x_dist)
            x_new,T_new = self.VLE(y[0])
            x.append(x_new)
            T.append(T_new)
            
            y_bot_op = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bot)
            y_top_op = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)
            
            #Rectifying section
            while (y_top_op[LK_index]/y_top_op[HK_index] < y_bot_op[LK_index]/y_bot_op[HK_index]):
                y.append((np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist))
                x_new,T_new = self.VLE(y[-1])
                x.append(x_new)
                T.append(T_new)
                
                if counter2 >100:
                    break
                counter2 += 1
                y_bot_op = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bot)
                y_top_op = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)
            
            #Feed plate
            N_feed = counter2
            
            #Stripping section
            while (np.array(x[-1][HK_index])<0.98*x_bot[HK_index] or np.array(x[-1][LK_index])>1.2*x_bot[LK_index]): #When reflux is specified and num_plates is not calculated x returned contains one extra evaluation, not true for case when num_plates is specified. This is why fudge factors are added
                y.append((np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bot))
                x_new,T_new = self.VLE(y[-1])
                x.append(x_new)
                T.append(T_new)
                if counter1 >100:
                    break
                counter1 += 1
                
            num_plates = len(y)-1 #Remove distillate stream, reboiler
            
        else: #Num plates specified
            #Calculate compositions
            x = np.zeros((num_plates+1, self.num_species))
            y = np.zeros((num_plates+1, self.num_species))
            T = np.zeros(num_plates+1)
            
            y[0] = x_dist
            x_new,T_new = self.VLE(y[0])
            x[0] = x_new
            T[0] = T_new
            
            if self.N_feed is None: #Feed plate not specified
                y_bot_op = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bot)
                y_top_op = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)
                flag_rect_section_present = 0
            
                for i in range (1, num_plates+1):
                    #Rectifying section
                    if (y_top_op[LK_index]/y_top_op[HK_index] < y_bot_op[LK_index]/y_bot_op[HK_index]):
                        
                        y[i] = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)
                        x_new,T_new = self.VLE(y[i])
                        x[i] = x_new
                        T[i] = T_new
                        
                        y_bot_op = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bot)
                        y_top_op = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)
                        N_feed = i #To get feed plate, only last value used, cant write outside if cause elif structure will be violated    
                        flag_rect_section_present = 1
                        
                    #Stripping section
                    elif (np.array(x[-1][HK_index])<0.98*x_bot[HK_index] or np.array(x[-1][LK_index])>1.2*x_bot[LK_index]):
                        y[i] = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bot)
                        x_new,T_new = self.VLE(y[i])
                        x[i] = x_new
                        T[i] = T_new
                        if not(flag_rect_section_present):
                            N_feed = num_plates
            else: #Feed plate specified
                N_feed = self.N_feed
                for i in range (1, num_plates+1):
                    #Rectifying section
                    if i < num_plates+1 - self.N_feed:
                        y[i] = (np.array(x_new)*Ln/Vn + (1-Ln/Vn)*x_dist)
                        x_new,T_new = self.VLE(y[i])
                        x[i] = x_new
                        T[i] = T_new
                        N_feed = self.N_feed
                    #Stripping section
                    else:
                        y[i] = (np.array(x_new)*Lm/Vm - (Lm/Vm-1)*x_bot)
                        x_new,T_new = self.VLE(y[i])
                        x[i] = x_new
                        T[i] = T_new
                                        
        self.N_feed = N_feed
        
        self.retrieve_results(num_plates, x, y, T, bot_flowrate, dist_flowrate, reflux, N_feed, x_dist, x_bot, self.min_reflux, self.N_min)
        return num_plates, x, y, T, bot_flowrate, dist_flowrate, reflux, N_feed, x_dist, x_bot, self.min_reflux, self.N_min
    
    def retrieve_results(self, num_plates, x, y, T, bot_flowrate, dist_flowrate, reflux, N_feed, x_dist, x_bot, min_reflux, N_min):
        
        dist_result = {'num_plates':num_plates, 'x':x.T, 'y':y.T, 'T':T, 
                       'bot_flowrate':bot_flowrate, 'dist_flowrate':dist_flowrate, 'reflux':reflux, 
                       'N_feed':N_feed, 'x_dist':x_dist, 'x_bot':x_bot, 'min_reflux':min_reflux, 'N_min':N_min
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
        
class DynamicDistillation():
    def __init__(self, name_species, col_P, q_feed, LK, HK, 
                 per_LK, per_HK, reflux=None, num_plates=None, 
                 holdup=None, gamma_model='ideal', N_feed=None):
        
        self.M_const = holdup
        self.num_plates = num_plates
        self.name_species = name_species
        self.reflux = reflux
        self.q_feed = q_feed
        self.col_P = col_P
        self.LK = LK
        self.HK = HK
        self.per_HK = per_HK
        self.per_LK = per_LK
        self.num_species = len(name_species)
        self.LK_index = name_species.index(LK)
        self.HK_index = name_species.index(HK)
        self.gamma_model= gamma_model
        self.N_feed=N_feed #Num plate from bottom
        
        self.nomenclature()
        self._Phases = None
        self._Inlet = None
        return
        
    def nomenclature(self):
        self.name_states = ['temp', 'mole_frac']
        self.names_states_out = ['temp', 'mole_frac']
        self.names_states_in = self.names_states_out
        #self.names_states_in.append('vol_flow')
        self.states_di = {
            'temp':{'dim':1, 'units': 'K'}, 
            'mole_frac':{'dim':len(self.name_species), 'index': self.name_species},
            }
        
    @property
    def Inlet(self):
        return self._Inlet

    @Inlet.setter
    def Inlet(self, inlet):
        self._Inlet = inlet
        self.feed_flowrate = inlet.mole_flow
        self.z_feed = inlet.mole_frac
        
        num_comp = self.num_species
        self.len_in = [1, num_comp]#, 1]
        self.len_out = [1, num_comp]
        states_in_dict = dict(zip(self.names_states_in, self.len_in))
        states_out_dict = dict(zip(self.names_states_out, self.len_out))
        
        self.states_in_dict = {'Inlet': states_in_dict}
        self.states_out_dict = {'Outlet': states_out_dict}
        self.column_startup()
    
    def get_inputs(self, time):
        inputs = get_inputs_new(time, self.Inlet, self.states_in_dict)
        return inputs
        
    def column_startup(self):
        #Total reflux conditions (Startup)
        ### Steady state values (Based on steady state column)
        column_user_inputs = {'name_species': self.name_species,
                              'col_P': self.col_P,  # Pa
                              'num_plates': self.num_plates, #exclude reboiler
                              'reflux': self.reflux, #L/D
                              'q_feed': self.q_feed, #Feed q value
                              'LK': self.LK, #LK
                              'HK': self.HK, #HK
                              'per_LK': self.per_LK,# % recovery LK in distillate
                              'per_HK': self.per_HK,# % recovery HK in distillate
                              'holdup': self.M_const,
                              'N_feed': self.N_feed
                              }
        steady_col = DistillationColumn(**column_user_inputs)
        steady_col.Inlet = self._Inlet
        (num_plates, x_ss, y_ss, T_ss, bot_flowrate, dist_flowrate, 
         reflux, N_feed, x_dist, x_bot, min_reflux, N_min) = steady_col.calc_plates(reflux = column_user_inputs['reflux'], 
                                                                                    num_plates = column_user_inputs['num_plates'])
        
        #Calculate compositions
        #For starup Total reflux
        #Total reflux
        x0 = np.zeros_like(x_ss)
        y0 = np.zeros_like(y_ss)
        T0 = np.zeros_like(T_ss)
        column_total_reflux = {'name_species': self.name_species,
                              'col_P': self.col_P,  # Pa
                              'num_plates': num_plates, #exclude reboiler
                              'reflux': 1000, #L/D
                              'q_feed': self.q_feed, #Feed q value
                              'LK': self.LK, #LK
                              'HK': self.HK, #HK
                              'per_LK': self.per_LK,# % recovery LK in distillate
                              'per_HK': self.per_HK,# % recovery HK in distillate
                              'holdup': self.M_const,
                              'N_feed': N_feed
                              }
        
        total_reflux_col = DistillationColumn(**column_total_reflux)
        total_reflux_col.Inlet = self._Inlet
        (num_plates, x0, y0, T0, bot_flowrate, dist_flowrate, 
          reflux, N_feed, x_dist, x_bot, min_reflux, N_min) = total_reflux_col.calc_plates(reflux = column_total_reflux['reflux'], 
                                                                                            num_plates = column_total_reflux['num_plates'])
        x0 = np.array(x0)
        y0 = np.array(y0)
        T0 = np.array(T0)
        T0 = (T0.T).ravel()
        ## Set all values to original steady state
        steady_col = DistillationColumn(**column_user_inputs)
        steady_col.Inlet = self._Inlet
        (num_plates, x_ss, y_ss, T_ss, bot_flowrate, dist_flowrate, 
          reflux, N_feed, x_dist, x_bot, min_reflux, N_min) = steady_col.calc_plates(reflux = column_user_inputs['reflux'], 
                                                                                    num_plates = column_user_inputs['num_plates'])
        self.x0 = x0
        self.y0 = y0
        self.T0 = T0
        self.x_ss = x_ss
        self.y_ss = y_ss
        self.T_ss = T_ss
        self.num_plates = num_plates
        self.bot_flowrate = bot_flowrate
        self.dist_flowrate = dist_flowrate
        self.reflux = reflux
        self.N_feed = N_feed
        self.x_dist = x_dist
        self.x_bot = x_bot
        #min_reflux and N_min are already available as self. 

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        classify_phases(self)
    
    def unit_model(self, time, states, d_states):

        '''This method will work by itself and does not need any user manipulation.
        Fill material and energy balances with your model.'''
        di_states = unpack_discretized(states, self.len_out,
                                       self.name_states)

        #states_reord = np.reshape(states, (self.num_plates + 1, self.len_states))
        #states_split = [states_reord[:,0], states_reord[:,1:]] #first column in temperature, others are compositions
        #dict_states = dict(zip(self.name_states, states_split))
        material = self.material_balances(time, **di_states)
        
        di_d_states = unpack_discretized(d_states, self.len_out,
                                       self.name_states)
        #d_states = np.reshape(d_states, ((self.num_plates + 1, self.len_states)))
        
        material[:,1:] = material[:,1:] - di_d_states['mole_frac'] #N_plates(N_components), only for compositions
        balances = material.ravel()
        return balances
    
    def material_balances(self, time, temp, mole_frac):
        x = mole_frac
        inputs = self.get_inputs(time)['Inlet']
        z_feed = inputs['mole_frac']
        ##GET STARTUP CONDITIONS
        (bot_flowrate, dist_flowrate, 
         reflux, N_feed, M_const) = (self.bot_flowrate, self.dist_flowrate, 
                                     self.reflux, self.N_feed, self.M_const)
        
        #CALCULATE COLUMN FLOWS
        #Rectifying section
        Ln  = reflux*dist_flowrate
        Vn  = Ln  + dist_flowrate
        #Stripping section
        Lm  = Ln + self.feed_flowrate + self.feed_flowrate*(self.q_feed-1) 
        Vm  = Lm  - bot_flowrate
        
        dxdt = np.zeros_like(x)
        p_vap = self._Inlet.AntoineEquation(temp)
        residuals_temp = (self.col_P - (x*p_vap).sum(axis=1))
        
        k_vals = self._Inlet.getKeqVLE(pres = self.col_P, temp=temp,
                                       x_liq = x)
        alpha = k_vals/k_vals[self.HK_index]
        
        y = ((alpha*x).T/np.sum(alpha*x,axis=1)).T
        
        #Rectifying section
        dxdt[0] = (1/M_const)*(Vn*y[1] + Ln*y[0] - Vn*y[0] - Ln*x[0]) #Distillate plate
        dxdt[1:N_feed-1] = (1/M_const)*(Vn*y[2:N_feed] + Ln*x[0:N_feed-2] - Vn*y[1:N_feed-1] - Ln*x[1:N_feed-1])
        #Stripping section
        dxdt[N_feed-1] = (1/M_const)*(Vm*y[N_feed] + Ln*x[N_feed-2] + self.feed_flowrate*z_feed - Vn*y[N_feed-1] - Lm*x[N_feed-1]) #Feed plate
        dxdt[N_feed:-1] = (1/M_const)*(Vm*y[N_feed+1:] + Lm*x[N_feed-1:-2] -Vm*y[N_feed:-1] - Lm*x[N_feed:-1])
        dxdt[-1] = (1/M_const)*(Vm*x[-1] + Lm*x[-2] -Vm*y[-1] - Lm*x[-1]) #Reboiler, y_in for reboiler is the same as x_out
        mat_bal = np.column_stack((residuals_temp, dxdt))
        return mat_bal
        
    def energy_balances(self, time, temp, mole_frac):
        pass
        return
    
    def solve_unit(self, runtime=None, t0=0):
        self.len_states = len(self.name_species) + 1 # 3 compositions + 1 temperature per plate
        
        init_states = np.column_stack((self.T0, self.x0))
        init_derivative = self.material_balances(time=0, mole_frac=self.x0, temp=self.T0)
        
        problem = Implicit_Problem(self.unit_model, init_states.ravel(), init_derivative.ravel(), t0)
        solver = IDA(problem)
        solver.rtol=10**-2
        solver.atol=10**-2
        time, states, d_states = solver.simulate(runtime)
        self.retrieve_results(time, states)
        return time, states, d_states

    def retrieve_results(self, time, states):
        time = np.asarray(time)
        self.timeProf = time
        
        indexes = {key: self.states_di[key].get('index', None)
                   for key in self.name_states}

        inputs = self.get_inputs(self.timeProf)['Inlet']

        dp = unpack_discretized(states, self.len_out, self.name_states,
                                indexes=indexes, inputs=inputs)

        dp['time'] = time
        
        self.result = DynamicResult(di_states=self.states_di, di_fstates=None, **dp)

        self.outputs = dp
        x_comp = np.array(list(dp['mole_frac'].values())) #[component_index, time, plate]
        
        # Outlet stream
        path = self.Inlet.path_data
        self.OutletBottom = LiquidStream(path, temp=dp['temp'][-1][-1], #[time, plate]
                                   mole_frac=x_comp.T[-1][-1], #[plate, time, component_index]
                                   mole_flow = self.bot_flowrate)
        self.OutletDistillate = LiquidStream(path, temp=dp['temp'][-1][0], #[time,plate]
                                   mole_frac=x_comp.T[0][-1],
                                   mole_flow = self.dist_flowrate)
        self.Outlet = self.OutletBottom