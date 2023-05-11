# -*- coding: utf-8 -*-


# import numpy as np
# from autograd import numpy as np
import numpy as np
from PharmaPy.ThermoModule import ThermoPhysicalManager
from PharmaPy.Commons import trapezoidal_rule
from scipy.optimize import newton

import warnings

eps = np.finfo(float).eps


def classify_phases(instance, names=None):

    phases = instance.Phases

    if names is None:
        solid_count = 1
        liquid_count = 1
        vapor_count = 1

        for phase in phases:
            if 'Liquid' in phase.__class__.__name__:
                phase_name = 'Liquid_{}'.format(liquid_count)
                liquid_count += 1

            elif 'Solid' in phase.__class__.__name__:
                phase_name = 'Solid_{}'.format(solid_count)
                solid_count += 1

            elif 'Vapor' in phase.__class__.__name__:
                phase_name = 'Vapor_{}'.format(solid_count)
                vapor_count += 1

            setattr(phase, 'name', phase_name)
            setattr(instance, phase_name, phase)
    else:
        for phase, name in zip(phases, names):
            setattr(phase, 'name', phase_name)
            setattr(instance, name, phase)


def getPropsPhaseMix(phases, basis='mass'):
    # Empty containers
    props_matrix = np.zeros((len(phases), 3))
    vfrac_phases = []
    props_matrix = []

    for ind, phase in enumerate(phases):
        all_props = phase.getProps(basis=basis)
        props_matrix.append(all_props[:3])

        if phase.__class__.__name__ == 'LiquidPhase':
            ind_liq = ind
        if phase.__class__.__name__ == 'SolidPhase':
            mom_solid = all_props[-2]
            conv_exp = np.arange(len(mom_solid))
            # num/m**3, m/m**3, m**2/m**3, ...
            mom_meters = mom_solid * (1e-6)**conv_exp

            vfrac_solid = mom_meters[-1] * phase.kv
            vfrac_phases.append(vfrac_solid)

    props_matrix = np.vstack(props_matrix)

    # Volume fraction and mass fractions of phases
    vfrac_rest = 1 - sum(vfrac_phases)
    vfrac_phases.insert(ind_liq, vfrac_rest)

    # to avoid internal casting next line
    vfrac_phases = np.array(vfrac_phases)

    mass_phases = vfrac_phases * props_matrix[:, 1]
    mfrac_phases = mass_phases / mass_phases.sum()

    cp, rho, enthalpy = props_matrix.T

    return cp, rho, enthalpy, vfrac_phases, mfrac_phases


class LiquidPhase(ThermoPhysicalManager):
    """ Creates a LiquidPhase object.
    
    Parameters
    ----------
    mass_frac : array-like (optional)
        mass fractions of the constituents of the phase.
    mole_conc : array-like (optional)
        molar concentrations of the constituents of the phase, excluding
        the solvent
    ind_solv : int
        index of solvent components in the liquid phase. It must be
        only specified if 'mass_frac' or 'mole_frac' are not given.
    """
    def __init__(self, path_thermo=None, temp=298.15, pres=101325,
                 mass=0, vol=0, moles=0,
                 mass_frac=None, mole_frac=None,
                 mass_conc=None, mole_conc=None,
                 name_solv=None, verbose=True, check_input=True):

        super().__init__(path_thermo)

        self.cp_liq = np.atleast_2d(self.cp_liq)
        self.p_vap = np.atleast_2d(self.p_vap)

        if name_solv is None:
            ind_solv = name_solv
        else:
            ind_solv = self.name_species.index(name_solv)

        self.ind_solv = ind_solv

        self.temp = float(temp)
        self.pres = pres

        self.mass = mass
        self.vol = vol
        self.moles = moles

        unspec_num = (mass_frac is None) + (mass_conc is None) + \
            (mole_conc is None) + (mole_frac is None)

        if unspec_num == 4:
            raise ValueError("No measure of composition was provided")
        elif unspec_num < 3:
            raise RuntimeWarning("More than one measure of composition was "
                                 "provided")

        if mass_frac is not None:
            self.mass_frac = np.array(mass_frac)
            self.mass_conc = mass_conc

            self.mole_frac = mole_frac
            self.mole_conc = mole_conc

            if self.mass_frac.ndim == 1:
                sum_fracs = sum(self.mass_frac)
                less_than_one = sum_fracs < 0.99
            else:
                sum_fracs = self.mass_frac.sum(axis=1)
                less_than_one = any(sum_fracs < 0.99)

            if less_than_one:
                if verbose:
                    print()
                    print('PharmaPy Warning: '
                          'The sum of mass fractions is less than 0.99 '
                          '(sum(mass_frac) = %.4f) for %s object'
                          % (sum_fracs, self.__class__.__name__))
                    print()

            self.__calcComposition()

        elif mole_frac is not None:
            self.mass_frac = mass_frac
            self.mass_conc = mass_conc

            self.mole_frac = np.array(mole_frac)
            self.mole_conc = mole_conc

            if self.mole_frac.ndim == 1:
                sum_fracs = sum(self.mole_frac)
                less_than_one = sum_fracs < 0.99
            else:
                sum_fracs = self.mole_frac.sum(axis=1)
                less_than_one = any(sum_fracs < 0.99)

            if less_than_one:
                if verbose:
                    print()
                    print('PharmaPy Warning: '
                          'The sum of mass fractions is less than 0.99 '
                          '(sum(mass_frac) = %.4f) for %s object'
                          % (sum_fracs, self.__class__.__name__))
                    print()

            self.__calcComposition()

        elif mass_conc is not None:
            self.mass_conc = np.array(mass_conc)
            self.mole_frac = mole_frac

            self.mass_frac = mass_frac
            self.mole_conc = mole_conc

            self.__calcComposition()

        elif mole_conc is not None:
            self.mole_conc = np.array(mole_conc)
            self.mole_frac = mole_frac

            self.mass_frac = mass_frac
            self.mass_conc = mass_conc

            self.__calcComposition()

        if (mass + vol + moles) == 0:
            if check_input:
                warnings.simplefilter("always")
                warnings.warn("'mass', 'moles' and 'vol' are all set to zero. "
                              "Model may not perform as intended.",
                              RuntimeWarning)

                warnings.simplefilter("ignore")

        self.y_upstream = None

        self._name = None
        self.transferred_from_uo = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def __set_amounts(self, mass, vol, moles, massfrac, molefrac,
                      conc, mass_conc):
        densMass = self.getDensityMix(massfrac)
        mw_av = np.dot(molefrac, self.mw)
        if mass > 0:
            self.mass = mass
            self.vol = mass / densMass
            self.moles = mass / mw_av * 1000

        elif vol > 0:
            self.vol = vol
            self.mass = vol * densMass
            self.moles = self.mass / mw_av * 1000

        elif moles > 0:
            self.moles = moles
            self.mass = moles * mw_av / 1000  # kg
            self.vol = self.mass / densMass

        self.mass_frac = massfrac
        self.mole_frac = molefrac
        self.mole_conc = conc
        self.mass_conc = mass_conc

        self.mw_av = mw_av

    def __calcComposition(self):

        if self.mole_conc is not None:
            frac_out = self.conc_to_frac(self.mole_conc,
                                         solvent_ind=self.ind_solv)
            if self.ind_solv is None:
                mass_frac, mole_frac = frac_out
                mole_conc = self.mole_conc
            else:
                mass_frac, mole_frac, mole_conc = frac_out

            mass_conc = mole_conc * self.mw  # kg / m3

        elif self.mass_conc is not None:
            frac_out = self.mass_conc_to_frac(self.mass_conc,
                                              solvent_ind=self.ind_solv)

            if self.ind_solv is None:
                mass_frac, mole_frac = frac_out
                mass_conc = self.mass_conc
            else:
                mass_frac, mole_frac, mass_conc = frac_out

            mole_conc = mass_conc / self.mw  # mol/L

        elif self.mass_frac is not None:
            mole_conc = self.frac_to_conc(self.mass_frac)
            mass_conc = mole_conc * self.mw

            mole_frac = self.frac_to_frac(self.mass_frac)
            mass_frac = self.mass_frac

        elif self.mole_frac is not None:
            mole_conc = self.frac_to_conc(mole_frac=self.mole_frac)
            mass_conc = mole_conc * self.mw

            mass_frac = self.frac_to_frac(mole_frac=self.mole_frac)
            mole_frac = self.mole_frac

        self.__set_amounts(self.mass, self.vol, self.moles,
                           mass_frac, mole_frac, mole_conc, mass_conc)

    def updatePhase(self, mole_conc=None, mass_conc=None,
                    mass_frac=None, mole_frac=None,
                    vol=0, mass=0, moles=0, temp=None, pres=None):

        if mole_conc is not None:
            frac_out = self.conc_to_frac(mole_conc,
                                         solvent_ind=self.ind_solv)
            if self.ind_solv:
                mass_frac, mole_frac, mole_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mass_conc = mole_conc * self.mw

        elif mass_conc is not None:
            frac_out = self.mass_conc_to_frac(mass_conc,
                                              solvent_ind=self.ind_solv)

            if self.ind_solv:
                mass_frac, mole_frac, mass_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mole_conc = mass_conc / self.mw

        elif mass_frac is not None:
            mole_conc = self.frac_to_conc(mass_frac)
            mass_conc = mole_conc * self.mw
            mole_frac = self.frac_to_frac(mass_frac)

        elif mole_frac is not None:
            mole_conc = self.frac_to_conc(mole_frac=mole_frac)
            mass_conc = mole_conc * self.mw
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)

        else:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac
            mole_conc = self.mole_conc
            mass_conc = self.mass_conc

        if temp is not None:
            self.temp = temp

        if pres is not None:
            self.pres = pres

        self.__set_amounts(mass, vol, moles, mass_frac, mole_frac,
                           mole_conc, mass_conc)

    def getDensity(self, mass_frac=None, mole_frac=None, temp=None,
                   basis='mass'):

        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        rhoLiq = self.getDensityMix(mass_frac, mole_frac, phase='liquid',
                                    basis=basis, temp=temp)

        return rhoLiq

    def getCp(self, temp=None, mass_frac=None, mole_frac=None, basis='mole'):
        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        cpLiq = super().getCpMix(temp, mass_frac, mole_frac, basis=basis)

        return cpLiq

    def getEnthalpy(self, temp=None, temp_ref=298.15, mass_frac=None,
                    mole_frac=None, total_h=True, basis='mass'):

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        if temp is None:
            temp = self.temp

        hLiq = super().getEnthalpy(temp, temp_ref, mass_frac, mole_frac,
                                   phase='liquid', total_h=total_h,
                                   basis=basis)

        return hLiq

    def getBubblePoint(self, pres=None, mass_frac=None, mole_frac=None,
                       thermo_method='ideal', y_vap=False):

        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac

        elif mole_frac is None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)

        if pres is None:
            pres = self.pres

        def bubble_fn(temp):
            k_vals = self.getKeqVLE(temp, pres, mole_frac,
                                    gamma_model=thermo_method)

            obj = np.dot(mole_frac, (k_vals - 1))

            return obj

        temp_pure = self.AntoineEquation(pres=pres)
        temp_seed = np.dot(mole_frac, temp_pure)
        temp_bubble = newton(bubble_fn, temp_seed, full_output=False)

        if y_vap:
            k_vals = self.getKeqVLE(temp_bubble, pres, mole_frac,
                                    gamma_model=thermo_method)

            y_frac = k_vals * mole_frac

            return temp_bubble, y_frac
        else:
            return temp_bubble

    def getBubblePressure(self, temp=None, mass_frac=None, mole_frac=None,
                          thermo_method='ideal', y_vap=False):

            if mass_frac is None and mole_frac is None:
                mole_frac = self.mole_frac

            elif mole_frac is None:
                mole_frac = self.frac_to_frac(mass_frac=mass_frac)

            if temp is None:
                temp = self.temp

            def bubble_fn(pr):
                k_vals = self.getKeqVLE(temp, pr, mole_frac,
                                        gamma_model=thermo_method)

                obj = np.dot(mole_frac, (k_vals - 1))

                return obj

            pres_pure = self.AntoineEquation(temp=temp)
            pres_seed = np.dot(mole_frac, pres_pure)
            pres_bubble = newton(bubble_fn, pres_seed, full_output=False)

            return pres_bubble

    def getProps(self, basis='mass'):
        cpmass, cpmole = self.getCpMix(self.temp, self.mass_frac)
        rhoMass, rhoMole = self.getDensityMix(self.mass_frac, temp=self.temp)
        hmass, hmole = self.getEnthalpy(self.temp, mass_frac=self.mass_frac)
        # viscosity = self.getViscosityMix(self.temp, self.mass_frac)
        if basis == 'mass':
            cp = cpmass
            enthalpy = hmass
            rho = rhoMass
        else:
            cp = cpmole
            enthalpy = hmole
            rho = rhoMole

        return cp, rho, enthalpy

    def getActivityCoeff(self, method='ideal', mole_frac=None, temp=None):

        if mole_frac is None:
            mole_frac = self.mole_frac

        if temp is None:
            temp = self.temp

        if method == 'ideal':
            gamma = np.ones_like(mole_frac)
        elif method == 'UNIQUAC':
            if 'qip' not in self.__dict__:
                self.qip = self.qi

            gamma = self.UNIQUAC(mole_frac, temp)

        else:
            gamma = self.UNIFAC_DMD(mole_frac, temp)

        return gamma

    def getViscosity(self, temp=None, mass_frac=None, mole_frac=None):
        viscosity = self.getViscosityMix(temp, mass_frac, mole_frac,
                                         phase='liquid')

        return viscosity

    def getSurfTensionPure(self, temp=None):
        surface_pure = self.surf_tension
        surface_pure[np.isnan(surface_pure)] = 0

        return surface_pure

    def getSurfTension(self, mass_frac=None, mole_frac=None, temp=None):

        if mass_frac is None:
            mass_frac = self.mass_frac

        if temp is None:
            temp = self.temp

        surfacePure = self.getSurfTensionPure(temp)
        surfaceMix = np.dot(mass_frac, surfacePure)

        return surfaceMix


class VaporPhase(ThermoPhysicalManager):
    def __init__(self, path_thermo=None, temp=298.15, pres=101325,
                 mass=0, vol=0, moles=0,
                 mass_frac=None, mole_frac=None, mole_conc=None,
                 check_input=True, verbose=True):

        super().__init__(path_thermo)

        # Calculate amount of material and compositions using LiquidPhase
        props = LiquidPhase(path_thermo, temp, pres, mass,
                            vol, moles, mass_frac, mole_frac, mole_conc,
                            check_input=check_input, verbose=verbose)

        self.mass = props.mass
        self.moles = props.moles
        self.vol = props.vol

        self.mass_frac = props.mass_frac
        self.mole_frac = props.mole_frac
        self.mole_conc = props.mole_conc

        self.mw_av = props.mw_av

        self.temp = float(temp)

        self.y_upstream = None
        self._name = None

        self.transferred_from_uo = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def __set_amounts(self, mass, vol, moles, massfrac, molefrac,
                      conc, mass_conc):
        densMass = self.getDensityMix(massfrac)
        mw_av = np.dot(molefrac, self.mw)
        if mass > 0:
            self.mass = mass
            self.vol = mass / densMass
            self.moles = mass / mw_av * 1000

        elif vol > 0:
            self.vol = vol
            self.mass = vol * densMass
            self.moles = self.mass / mw_av * 1000

        elif moles > 0:
            self.moles = moles
            self.mass = moles * mw_av / 1000  # kg
            self.vol = self.mass / densMass

        self.mass_frac = massfrac
        self.mole_frac = molefrac
        self.mole_conc = conc
        self.mass_conc = mass_conc

        self.mw_av = mw_av

    def updatePhase(self, mole_conc=None, mass_conc=None,
                    mass_frac=None, mole_frac=None,
                    vol=0, mass=0, moles=0):

        if mole_conc is not None:
            frac_out = self.conc_to_frac(mole_conc,
                                         solv_ind=self.ind_solv)
            if self.ind_solv:
                mass_frac, mole_frac, mole_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mass_conc = mole_conc * self.mw

        elif mass_conc is not None:
            frac_out = self.mass_conc_to_frac(mass_conc,
                                              solv_ind=self.ind_solv)

            if self.ind_solv:
                mass_frac, mole_frac, mass_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mole_conc = mass_conc / self.mw

        elif mass_frac is not None:
            mole_conc = self.frac_to_conc(mass_frac)
            mass_conc = mole_conc * self.mw
            mole_frac = self.frac_to_frac(mass_frac)

        elif mole_frac is not None:
            mole_conc = self.frac_to_conc(mole_frac=mole_frac)
            mass_conc = mole_conc * self.mw
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)

        else:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac
            mole_conc = self.mole_conc
            mass_conc = self.mass_conc

        self.__set_amounts(mass, vol, moles, mass_frac, mole_frac,
                           mole_conc, mass_conc)

    def getCp(self, temp, mass_frac=None, mole_frac=None, basis='mass'):
        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac

        cpMix = self.getCpMix(temp, mass_frac, mole_frac, phase='vapor',
                              basis=basis)

        return cpMix

    def getHeatVaporization(self, temp, basis='mass'):
        # if idx is None:
        #     idx = np.arange(len(self.t_crit))

        temp = np.atleast_1d(temp)
        num_comp = len(self.t_crit)

        num_temp = len(temp)
        if num_temp > 1:
            temp = temp[..., np.newaxis]
            idx = np.unique(np.where(temp < self.t_crit)[1])
            delta_shape = (num_temp, num_comp)
        else:
            idx = np.where(temp < self.t_crit)[0]
            delta_shape = num_comp

        tref = self.tref_hvap[idx]

        watson = ((self.t_crit[idx] - temp) / (self.t_crit[idx] - tref))**0.38

        deltahvap = np.zeros(delta_shape)

        if num_temp > 1:
            deltahvap[:, idx] = (watson * self.delta_hvap[idx])  # J/mole
        else:
            deltahvap[idx] = (watson * self.delta_hvap[idx])  # J/mole

        if basis == 'mass':
            if num_temp > 1:
                deltahvap = deltahvap[:, idx] / self.mw[idx] * 1000  # J/kg
            else:
                deltahvap = deltahvap[idx] / self.mw[idx] * 1000  # J/kg

        return deltahvap

    def getEnthalpy(self, temp=None, temp_ref=298.15, mass_frac=None,
                    mole_frac=None, total_h=True, basis='mass'):
        """ Calculate vapor phase enthalpy. It assumes that the reference state
        is a liquid at t_ref.

        Parameters
        ----------
        temp : float or array-like
            Temperature for enthalpy calculation in K.   
        temp_ref : float, optional
            Reference temperature for enthalpy calculation. The default is 298.15.
        mass_frac : array-like, optional
            Fraction of the species participating in the vapor phase in mass. The default is None.
        mole_frac : array-like, optional
            Fraction of the species participating in the vapor phase in mole. The default is None.
        total_h : bool, optional
            If True, the total enthalpy is returned. If False, an array
            of individual enthalpy for each species is returned.
            he default is True.

        Returns
        -------
        hvapMass : J/kg
        hvapMole : J/mol

        """
        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac
        elif mass_frac is None:
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)
        else:
            mole_frac = self.frac_to_frac(mass_frac)

        if temp is None:
            temp = self.temp

        # Sensible heat
        if any(temp > self.t_crit):
            ind_super = np.where(temp > self.t_crit)[0]
            ind_sub = np.where(temp < self.t_crit)[0]

            ind_sort = np.argsort(np.concatenate((ind_super, ind_sub)))

            sensSuper = super().getEnthalpy(
                temp, temp_ref, mass_frac, mole_frac, total_h=total_h,
                idx=ind_super, phase='vapor', basis=basis)

            if len(ind_sub) > 0:
                sensSub = super().getEnthalpy(
                    temp, temp_ref, mass_frac, mole_frac, phase='liquid',
                    total_h=total_h, idx=ind_sub, basis=basis)

                if total_h:
                    hSens = sensSuper + sensSub
                else:
                    hSens = np.concatenate((sensSuper, sensSub))[ind_sort]

            else:
                hSens = sensSuper

        else:
            hSens = super().getEnthalpy(
                temp, temp_ref, mass_frac, mole_frac, phase='liquid',
                total_h=total_h, basis=basis)

        # Phase change
        deltaVap = self.getHeatVaporization(temp, basis=basis)

        # Collect terms
        if total_h:
            hVap = hSens + np.dot(deltaVap, mole_frac)

        else:
            hVap = hSens + deltaVap

        return hVap

    def AntoineEquation(self, temp=None, pres=None):
        a_ct, b_ct, c_ct = self.p_vap.T

        if pres is None:
            if isinstance(temp, np.ndarray):
                temp = temp[..., np.newaxis]

            vap_pressure = a_ct - b_ct / (temp + c_ct)

            return 10**(vap_pressure)

        else:
            if isinstance(pres, np.ndarray):
                pres = pres[..., np.newaxis]

            temp_sat = b_ct / (a_ct - np.log10(pres)) - c_ct

            return temp_sat

    def getDewPoint(self, pres=None, mass_frac=None, mole_frac=None,
                    thermo_method='ideal', x_liq=False):

        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac

        elif mole_frac is None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)

        if pres is None:
            pres = self.pres

        def dew_fn(temp):
            k_vals = self.getKeqVLE(temp, pres, mole_frac,
                                    gamma_model=thermo_method)

            obj = np.dot(mole_frac, 1/k_vals) - 1

            return obj
        temp_pure = self.AntoineEquation(pres=pres)
        temp_seed = np.dot(mole_frac, temp_pure)
        temp_dew = newton(dew_fn, temp_seed, full_output=False)

        if x_liq:
            k_vals = self.getKeqVLE(temp_dew, pres, mole_frac,
                                    gamma_model=thermo_method)

            x_frac = mole_frac/k_vals

            return temp_dew, x_frac
        else:
            return temp_dew

    def getViscosity(self, temp=None, mass_frac=None, mole_frac=None):
        viscosity = self.getViscosityMix(temp, mass_frac, mole_frac,
                                         phase='vapor')

        return viscosity

    def getDensity(self, pres_gas=None, temp_gas=None, phase ='gas', basis='mole'):

        if pres_gas is None and temp_gas is None:

            pres_gas = self.pres_gas
            temp_gas = self.temp

        densGas = pres_gas/ (8.314 * temp_gas)

        return densGas


class SolidPhase(ThermoPhysicalManager):
    """    

    Parameters
    ----------
    path_thermo : string
        Directory of the physical properties .json file
    temp : float or array-like
        Temperature for enthalpy calculation in K.   
    temp_ref : float, optional
        Reference temperature for enthalpy calculation. The default is 298.15.
    pres : float, optional
        Pressure in atmosphere of the system in pascals. The default is 101325.
    mass : float, optional
        Mass of solids in kg. The default is 0.
    mass_frac : array-like, optional
        Fraction of the species participating in the solid phase in mass basis.
        The default is None.
    moments : array, optional
        Array of size N, containing the distribution moments in um**n, 
        for n = 0,...,N - 1. The default is None.
    num_mom : integer, optional
        Maximum order of moments describing solid phase. The default is 4.
    x_distrib : array, optional
        Array of size N, containing the internal grid
        size coordinate of the solids [um]. The default is None
    distrib : array, optional
        Array of size N, constaining the initial distribution of crystals
        [#/m**3/um]. The default is None.
    distrib_type : string, optional
        Type of distribution of crystals. The option is 'mass_frac' 
        or 'vol_perc'. The default is 'vol_perc'.
    moisture : float, optional
        Initial moisture content of the solids. The default is 0.
    porosity : float, optional
        Volume-based pore fraction out of the packed solid beds. The default is 0.
    mole_conc : array-like, optional
        Concentration of the species participating in the solid phase in mole basis. The default is None.
    kv : float, optional
        Volumetric shape factor of the solids. The default is 1.



    Returns
    -------
    None.

    """
    
    def __init__(self, path_thermo, temp=298.15, temp_ref=298.15, pres=101325,
                 mass=0, mass_frac=None,
                 moments=None, num_mom=4,
                 distrib=None, x_distrib=None, distrib_type='vol_perc',
                 moisture=0, porosity=0,
                 mole_conc=None, kv=1):
        
        super().__init__(path_thermo)
        self.kv = kv
        self.distrib_type = distrib_type

        self.cp_solid = np.atleast_2d(self.cp_solid)

        self.temp = temp
        self.temp_ref = temp_ref
        self.pres = pres

        self.mass = mass

        mass_frac = np.atleast_1d(mass_frac)
        mass_frac[mass_frac == 0] = eps

        self.mass_frac = mass_frac
        self.mole_frac = self.frac_to_frac(mass_frac=self.mass_frac)

        solid_spec = False

        if moments is not None:
            self.num_mom = len(moments)
            self.moments = moments

            self.x_distrib = x_distrib
            self.distrib = distrib

            solid_spec = True

        elif distrib is not None:
            x_distrib = np.asarray(x_distrib)

            self.x_distrib = x_distrib
            self.distrib = self.getDistribution(x_distrib, distrib)

            self.num_distrib = len(distrib)
            self.num_mom = num_mom

            mom_idx = np.arange(self.num_mom)
            self.moments = self.getMoments(mom_num=mom_idx)

            solid_spec = True

        else:
            pass
            # print('Neither moment nor distribution data was '
            #       'provided for this SolidPhase object. Make sure to provide '
            #       'one of the two either when declaring this phase, or in a '
            #       'Slurry object to which this phase is aggregated')

        # Mass and volume
        dens = self.getDensity()

        if solid_spec:
            if self.mass == 0:
                self.vol = self.moments[3] * kv
                self.mass = self.vol * dens
            else:
                self.vol = self.mass / dens

        mw_av = np.dot(self.mole_frac, self.mw)
        self.moles = mass / mw_av

        if mass_frac is not None:
            sum_fracs = sum(mass_frac)
            if sum_fracs < 0.99:
                raise RuntimeError(
                    'The sum of mass fractions is less than 0.99')

        self.moisture = moisture
        self.porosity = porosity
        self.distribProf = None

        self._name = None
        self.transferred_from_uo = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def updatePhase(self, x_distrib=None, distrib=None, mass=None,
                    moments=None):
        if x_distrib is not None:
            self.x_distrib = x_distrib

        if distrib is not None:
            self.distrib = distrib
            self.moments = self.getMoments()
            self.num_distrib = len(distrib)

            self.vol = self.moments[3]
            self.mass = self.moments[3] * self.getDensity()

        if mass is not None:
            self.mass = mass
            self.vol = mass / self.getDensity()

        if moments is not None:
            self.moments = moments

    def convert_distribution(self, x_distrib=None, num_distr=None,
                             vol_distr=None, mass=0):
        if x_distrib is None:
            x_distrib = self.x_distrib

        if num_distr is not None and vol_distr is not None:
            raise ValueError("Specify either 'num_distr' or 'vol_distr', "
                             "not both")
        elif num_distr is not None:  # convert to vol perc
            mom_three = self.getMoments(distrib=num_distr, mom_num=3)
            mom_three[mom_three == 0] = eps

            distrib_out = num_distr * self.dx * x_distrib**3 * self.kv / \
                mom_three / 1e18
        elif vol_distr is not None:
            if mass == 0:
                raise ValueError("'vol_perc' given, mass must be greater "
                                 "than zero.")
            dens = self.getDensity()
            distrib_out = (mass / dens) * vol_distr / self.kv / \
                x_distrib**3 / self.dx * 1e18  # number/um

        return distrib_out

    def getDistribution(self, x_distrib, distrib):
        dens = self.getDensity()

        # Crystal size dimension
        delta_x = np.diff(x_distrib)
        equal = np.isclose(delta_x[1:], delta_x[:-1]).all()
        if equal:
            self.dx = delta_x[0]
        else:  # assume geometric series and make adjustments
            ratio = x_distrib[1] / x_distrib[0]
            x_shifted = np.zeros(len(x_distrib) + 1)
            x_gr = np.sqrt(x_distrib[1:] * x_distrib[:-1])

            x_shifted[0] = x_gr[0] / ratio
            x_shifted[-1] = x_gr[-1] * ratio

            x_shifted[1:-1] = x_gr

            self.dx = np.diff(x_shifted)

        # Distribution
        distrib = np.asarray(distrib)
        if self.mass > 0:
            distrib = distrib / distrib.sum()
            if self.distrib_type == 'vol_perc':
                distr = self.convert_distribution(vol_distr=distrib,
                                                  mass=self.mass)
            elif self.distrib_type == 'mass_perc':
                distr = self.mass*distrib / x_distrib**3 / self.kv * 1e18

        else:
            distr = distrib

        return distr

    def getMoments(self, x_distrib=None, distrib=None, mom_num=None):
        if x_distrib is None:
            x_distrib = self.x_distrib

        if distrib is None:
            distrib = self.distrib

        if mom_num is None:
            mom_ind = range(4)
        elif isinstance(mom_num, int):
            mom_ind = [mom_num]
        else:
            mom_ind = mom_num

        if distrib.ndim == 1 or len(distrib) == 1:
            moments = np.zeros(len(mom_ind))
            for ind, exp in enumerate(mom_ind):
                integrand = distrib * x_distrib**exp
                moments[ind] = trapezoidal_rule(x_distrib, integrand.T)

            if len(mom_ind) == 1:
                moments = moments[0]

        else:
            moments = np.zeros((len(distrib), len(mom_ind)))
            for ind, exp in enumerate(mom_ind):
                integrand = distrib * x_distrib**exp
                moments[:, ind] = trapezoidal_rule(x_distrib, integrand.T)

        conv_factors = (1e-6)**np.array(mom_ind)
        moments *= conv_factors

        return moments

    def getDensity(self, mass_frac=None, mole_frac=None, temp=None,
                   basis='mass'):

        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            # mole_frac = self.mole_frac

        densSolid = self.getDensityMix(mass_frac, mole_frac, phase='solid',
                                       temp=temp, basis=basis)

        return densSolid

    def getPorosity(self, distrib=None, diam_filter=1, AR=None,
                    sphericity=None):

        if distrib is None:
            distrib = self.distrib
            mom_zero = self.moments[0]
            mom_one = self.moments[1]
        else:
            mom_zero, mom_one = self.getMoments(mom_num=(0, 1))

        # mom_one *= 1e-6  # m
        x_dist = self.x_distrib * 1e-6  # m

        if AR is None:
            AR = 2

        if sphericity is None:
            sphericity = 0.7

        # Yu, Zou et al (1996) and Yu,Zou, Stnadish (1996) model
        kv = 0.524  # Volumetric shape coefficient
        ks = 3.142  # Surface shape coefficient

        del_x_dist = np.diff(x_dist)
        node_x_dist = (x_dist[:-1] + x_dist[1:]) / 2
        node_CSD = (distrib[:-1] + distrib[1:]) / 2

        # Volume of crystals in each bin
        vol_cry = node_CSD * del_x_dist * (kv * node_x_dist**3)
        frac_vol_cry = vol_cry / (np.sum(vol_cry) + eps)

        vol_particle = kv * node_x_dist**3
        d_part_sphere = (6 * vol_particle / np.pi)**(1/3)
        d_part_equiv_pack = d_part_sphere / (sphericity**2.785 *
                                             np.exp(2.946 * (1 - sphericity)))

        # Initial porosity
        D_mean = mom_one/(mom_zero + eps)
        E_0_Jeschar = 0.375 + 0.34 * D_mean/diam_filter  # average porosity of packing of uniform sized spheres [-]

        initial_porosity = E_0_Jeschar

        V = 1/(1 - initial_porosity) * np.ones_like(node_x_dist)  # Specific Volume for initial porosity

        # Evaluate specific volume using modified linear packing model
        num_x = len(node_x_dist)
        V_T_node = np.zeros(num_x)

        for i in range(num_x):

            r = d_part_equiv_pack[:i] / d_part_equiv_pack[i]
            g_r = (1 - r)**2 + 0.4*r*(1 - r)**3.7
            V_large_j = V[:i] - (V[:i] - 1) * g_r - V[i]
            sum_V_large_term = sum(V_large_j * frac_vol_cry[:i])

            r_inv = r = d_part_equiv_pack[i] / d_part_equiv_pack[i + 1:]
            f_r = (1 - r_inv)**3.3 + 2.8*r_inv*(1 - r_inv)**2.7
            V_small_j = V[i + 1:] * (1 - f_r) - V[i]
            sum_V_small_term = sum(V_small_j * frac_vol_cry[i + 1:])

            V_T_node[i] = V[i] + sum_V_large_term + sum_V_small_term

        V_T = max(V_T_node)

        porosity = 1 - 1/V_T

        return porosity

    # def getPorosity(self, distrib=None, diam_filter=1):  # x_distrib is the x
    #     if distrib is None:
    #         distrib = self.distrib
    #         mom_zero = self.moments[0]
    #         mom_one = self.moments[1]
    #     else:
    #         mom_zero, mom_one = self.getMoments(mom_num=(0, 1))

    #     # mom_one *= 1e-6  # m
    #     x_dist = self.x_distrib * 1e-6  # m

    #     # Ouchiyiama model
    #     E_denom = np.zeros_like(x_dist)

    #     for p in range(4, len(E_denom)):
    #         xx = x_dist[:p]
    #         CSD = distrib[:p]

    #         D_mean = mom_one / mom_zero

    #         # average porosity of packing of uniform sized spheres [-]
    #         E_0_Jeschar = 0.375 + 0.34 * D_mean / diam_filter

    #         DD = xx - D_mean
    #         DD[DD <= 0] = 0

    #         # n value
    #         n_num = np.dot((xx + D_mean)**2,
    #                        (1 - 3/8*(D_mean/(xx + D_mean)))*CSD)

    #         n_denom = np.dot((xx**3 - DD**3), CSD)

    #         n_bar = 1 + 4/13*D_mean*(7 - 8*E_0_Jeschar)*(n_num/n_denom)

    #         E_denom[p] = np.dot((DD**3 + (1/n_bar)*((xx + D_mean)**3 - DD**3)),
    #                             CSD)

    #     E_denom = max(E_denom)
    #     E_num = np.dot(x_dist**3, distrib)

    #     porosity = max(0, 1 - E_num/E_denom)

    #     return porosity

    def getCp(self, temp=None, mass_frac=None, mole_frac=None, basis='mass'):
        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        cpSolid = super().getCpMix(temp, mass_frac, mole_frac, phase='solid',
                                   basis=basis)

        return cpSolid

    def getEnthalpy(self, temp=None, temp_ref=298.15, mass_frac=None,
                    mole_frac=None, total_h=True, basis='mass'):

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        if temp is None:
            temp = self.temp

        hSolid = super().getEnthalpy(temp, temp_ref, mass_frac, mole_frac,
                                     phase='solid', total_h=total_h,
                                     basis=basis)

        return hSolid
