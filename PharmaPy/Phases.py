# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:55:49 2020

@author: dcasasor
"""

# import numpy as np
# from autograd import numpy as np
import numpy as np
from PharmaPy.ThermoModule import ThermoPhysicalManager
from PharmaPy.Commons import trapezoidal_rule
from scipy.optimize import newton

eps = np.finfo(float).eps


def classify_phases(instance, names=None):

    phases = instance.Phases

    if names is None:
        solid_count = 1
        liquid_count = 1
        vapor_count = 1

        for phase in phases:
            if 'Liquid' in phase.__class__.__name__:
                setattr(instance, 'Liquid_{}'.format(liquid_count), phase)
                liquid_count += 1
            elif 'Solid' in phase.__class__.__name__:
                setattr(instance, 'Solid_{}'.format(solid_count), phase)
                solid_count += 1
            elif 'Vapor' in phase.__class__.__name__:
                setattr(instance, 'Vapor_{}'.format(solid_count), phase)
                vapor_count += 1
    else:
        for phase, name in zip(phases, names):
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
            mom_meters = mom_solid * (1e-6)**conv_exp  # num/m**3, m/m**3, m**2/m**3, ...

            vfrac_solid = mom_meters[-1] * phase.kv
            vfrac_phases.append(vfrac_solid)

    props_matrix = np.vstack(props_matrix)

    # Volume fraction and mass fractions of phases
    vfrac_rest = 1 - sum(vfrac_phases)
    vfrac_phases.insert(ind_liq, vfrac_rest)

    vfrac_phases = np.array(vfrac_phases)  # to avoid internal casting next line

    mass_phases = vfrac_phases * props_matrix[:, 1]
    mfrac_phases = mass_phases / mass_phases.sum()

    cp, rho, enthalpy = props_matrix.T

    return cp, rho, enthalpy, vfrac_phases, mfrac_phases


class LiquidPhase(ThermoPhysicalManager):
    def __init__(self, path_thermo=None, temp=298.15, pres=101325,
                 mass=0, vol=0, moles=0,
                 mass_frac=None, mole_frac=None,
                 mass_conc=None, mole_conc=None,
                 ind_solv=None, verbose=True):

        super().__init__(path_thermo)

        """ Creates a LiquidPhase object
        Parameters
        ----------
        mass_frac : array-like (optional)
            mass fractions of the constituents of the phase.
        concentr : array-like (optional)
            molar concentrations of the constituents of the phase, excluding
            the solvent
        num_comp : int
            number of components in the phase. It must be specified only if
            'mass_frac' or 'concentr' are not given.
        ind_solv : int
            index of solvent components in the liquid phase. It must be 
            only specified if 'mass_frac' or 'mole_frac' are not given.
        """

        self.cp_liq = np.atleast_2d(self.cp_liq)
        self.p_vap = np.atleast_2d(self.p_vap)
        self.ind_solv = ind_solv

        self.temp = np.float(temp)
        self.pres = pres

        self.mass = mass
        self.vol = vol
        self.moles = moles

        if mass_frac is None and mass_conc is None and mole_conc is None and mole_frac is None:
            self.mass_frac = np.ones(self.num_species) * eps
            self.mass_conc = np.ones(self.num_species) * eps

            self.mole_frac = np.ones(self.num_species) * eps
            self.concentr = mole_conc

            self.mw_av = 0
        elif mass_frac is not None:
            self.mass_frac = np.array(mass_frac)
            self.mass_conc = mass_conc

            self.mole_frac = mole_frac
            self.concentr = mole_conc

            if self.mass_frac.ndim == 1:
                sum_fracs = sum(self.mass_frac)
                less_than_one = sum_fracs < 0.99
            else:
                sum_fracs = self.mass_frac.sum(axis=1)
                less_than_one = any(sum_fracs < 0.99)

            if less_than_one:
                if verbose:
                    print()
                    print('The sum of mass fractions sum is less than 0.99')
                    print(mass_frac)
                    print()

            self.__calcComposition()

        elif mole_frac is not None:
            self.mass_frac = mass_frac
            self.mass_conc = mass_conc

            self.mole_frac = np.array(mole_frac)
            self.concentr = mole_conc
            sum_fracs = sum(mole_frac)
            if verbose:
                if sum_fracs < 0.99:
                    print()
                    print('The sum of mass fractions sum is less than 0.99')
                    print(mole_frac)
                    print()

            self.__calcComposition()

        elif mass_conc is not None:
            self.mass_conc = np.array(mass_conc)
            self.mole_frac = mole_frac

            self.mass_frac = mass_frac
            self.concentr = mole_conc

            self.__calcComposition()

        elif mole_conc is not None:
            self.concentr = np.array(mole_conc)
            self.mole_frac = mole_frac

            self.mass_frac = mass_frac
            self.mass_conc = mass_conc

            self.__calcComposition()

        self.y_upstream = None

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
        self.concentr = conc
        self.mass_conc = mass_conc

        self.mw_av = mw_av

    def __calcComposition(self):

        if self.concentr is not None:
            frac_out = self.conc_to_frac(self.concentr,
                                         solvent_ind=self.ind_solv)
            if self.ind_solv is None:
                mass_frac, mole_frac = frac_out
                concentr = self.concentr
            else:
                mass_frac, mole_frac, concentr = frac_out

            mass_conc = concentr * self.mw  # kg / m3

        elif self.mass_conc is not None:
            frac_out = self.mass_conc_to_frac(self.mass_conc,
                                              solvent_ind=self.ind_solv)

            if self.ind_solv is None:
                mass_frac, mole_frac = frac_out
                mass_conc = self.mass_conc
            else:
                mass_frac, mole_frac, mass_conc = frac_out

            concentr = mass_conc / self.mw  # mol/L

        elif self.mass_frac is not None:
            concentr = self.frac_to_conc(self.mass_frac)
            mass_conc = concentr * self.mw

            mole_frac = self.frac_to_frac(self.mass_frac)
            mass_frac = self.mass_frac

        elif self.mole_frac is not None:
            concentr = self.frac_to_conc(mole_frac=self.mole_frac)
            mass_conc = concentr * self.mw

            mass_frac = self.frac_to_frac(mole_frac=self.mole_frac)
            mole_frac = self.mole_frac

        self.__set_amounts(self.mass, self.vol, self.moles,
                           mass_frac, mole_frac, concentr, mass_conc)

    def updatePhase(self, concentr=None, mass_conc=None,
                    mass_frac=None, mole_frac=None,
                    vol=0, mass=0, moles=0, temp=None, pres=None):

        if concentr is not None:
            frac_out = self.conc_to_frac(concentr,
                                         solvent_ind=self.ind_solv)
            if self.ind_solv:
                mass_frac, mole_frac, concentr = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mass_conc = concentr * self.mw

        elif mass_conc is not None:
            frac_out = self.mass_conc_to_frac(mass_conc,
                                              solvent_ind=self.ind_solv)

            if self.ind_solv:
                mass_frac, mole_frac, mass_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            concentr = mass_conc / self.mw

        elif mass_frac is not None:
            concentr = self.frac_to_conc(mass_frac)
            mass_conc = concentr * self.mw
            mole_frac = self.frac_to_frac(mass_frac)

        elif mole_frac is not None:
            concentr = self.frac_to_conc(mole_frac=mole_frac)
            mass_conc = concentr * self.mw
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)

        else:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac
            concentr = self.concentr
            mass_conc = self.mass_conc

        if temp is not None:
            self.temp = temp

        if pres is None:
            self.pres = pres

        self.__set_amounts(mass, vol, moles, mass_frac, mole_frac,
                           concentr, mass_conc)

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
                 mass_frac=None, mole_frac=None, concentr=None):

        super().__init__(path_thermo)

        # Calculate amount of material and compositions using LiquidPhase
        props = LiquidPhase(path_thermo, temp, pres, mass,
                            vol, moles, mass_frac, mole_frac, concentr)

        self.mass = props.mass
        self.moles = props.moles
        self.vol = props.vol

        self.mass_frac = props.mass_frac
        self.mole_frac = props.mole_frac
        self.concentr = props.concentr

        self.mw_av = props.mw_av

        self.temp = float(temp)

        self.y_upstream = None

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
        self.concentr = conc
        self.mass_conc = mass_conc

        self.mw_av = mw_av

    def updatePhase(self, concentr=None, mass_conc=None,
                    mass_frac=None, mole_frac=None,
                    vol=0, mass=0, moles=0):

        if concentr is not None:
            frac_out = self.conc_to_frac(concentr,
                                         solv_ind=self.ind_solv)
            if self.ind_solv:
                mass_frac, mole_frac, concentr = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mass_conc = concentr * self.mw

        elif mass_conc is not None:
            frac_out = self.mass_conc_to_frac(mass_conc,
                                              solv_ind=self.ind_solv)

            if self.ind_solv:
                mass_frac, mole_frac, mass_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            concentr = mass_conc / self.mw

        elif mass_frac is not None:
            concentr = self.frac_to_conc(mass_frac)
            mass_conc = concentr * self.mw
            mole_frac = self.frac_to_frac(mass_frac)

        elif mole_frac is not None:
            concentr = self.frac_to_conc(mole_frac=mole_frac)
            mass_conc = concentr * self.mw
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)

        else:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac
            concentr = self.concentr
            mass_conc = self.mass_conc

        self.__set_amounts(mass, vol, moles, mass_frac, mole_frac,
                           concentr, mass_conc)

    def getCp(self, temp, mass_frac=None, mole_frac=None, basis='mass'):
        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac

        cpMix = self.getCpMix(temp, mass_frac, mole_frac, phase='vapor',
                              basis=basis)

        return cpMix

    def getHeatVaporization(self, temp, idx=None, basis='mass'):
        if idx is None:
            idx = np.arange(len(self.t_crit))

        temp = np.atleast_1d(temp)
        tref = self.tref_hvap[idx]

        if len(temp) > 1:
            temp = temp[..., np.newaxis]

        watson = ((self.t_crit[idx] - temp) / (self.t_crit[idx] - tref))**0.38

        deltahvap = watson * self.delta_hvap[idx]  # J/mole

        if basis == 'mass':
            deltahvap = deltahvap / self.mw[idx] * 1000  # J/kg

        return deltahvap

    def getEnthalpy(self, temp=None, temp_ref=298.15, mass_frac=None,
                    mole_frac=None, total_h=True, basis='mass'):
        """
        Calculate vapor phase enthalpy. It assumes that the reference state
        is a liquid at t_ref

        Parameters
        ----------
        temp : float or array-like
            DESCRIPTION.
        temp_ref : float, optional
            DESCRIPTION. The default is 298.15.
        mass_frac : array-like, optional
            DESCRIPTION. The default is None.
        mole_frac : array-like, optional
            DESCRIPTION. The default is None.
        total_h : bool, optional
            If True, the total enthalpy is returned. If False, an array
            of individual enthalpy for each species is returned.
            he default is True.

        Returns
        -------
        hvapMass : TYPE
            DESCRIPTION.
        hvapMole : TYPE
            DESCRIPTION.

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

                deltaVap = self.getHeatVaporization(
                    temp, idx=ind_sub, basis=basis)

                deltaVap = np.concatenate(
                    (np.zeros_like(ind_super), deltaVap))[ind_sort]

                if total_h:
                    hSens = sensSuper + sensSub
                else:
                    hSens = np.concatenate((sensSuper, sensSub))[ind_sort]

            else:
                hSens = sensSuper
                deltaVap = np.zeros_like(ind_super)

        else:
            hSens = super().getEnthalpy(
                temp, temp_ref, mass_frac, mole_frac, phase='liquid',
                total_h=total_h)

            deltaVap = self.getHeatVaporization(temp, basis=basis)

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
                    thermo_method='ideal', y_vap=False):

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
    def __init__(self, path_thermo, temp=298.15, temp_ref=298.15, pres=101325,
                 mass=0, mass_frac=None,
                 moments=None,
                 distrib=None, x_distrib=None, distrib_type='vol_perc',
                 moisture=0, porosity=0,
                 concentr=None, kv=1):

        super().__init__(path_thermo)
        self.kv = kv

        self.cp_solid = np.atleast_2d(self.cp_solid)

        self.temp = temp
        self.temp_ref = temp_ref
        self.pres = pres

        self.moments = moments
        self.distrib = distrib
        self.x_distrib = x_distrib
        self.num_mom = None

        mass_frac = np.atleast_1d(mass_frac)
        mass_frac[mass_frac == 0] = eps
        self.mass_frac = mass_frac
        self.mole_frac = self.frac_to_frac(mass_frac=self.mass_frac)

        if moments is not None:
            self.num_mom = len(moments)

        dens = self.getDensity()
        if distrib is not None:
            x_distrib = np.asarray(x_distrib)
            self.x_distrib = x_distrib

            delta_x = np.diff(x_distrib)
            equal = np.isclose(delta_x[1:], delta_x[:-1]).all()
            if equal:
                self.dx = delta_x[0]
            else:  # assume geometric series and make adjustments
                ratio = x_distrib[1] / x_distrib[0]
                x_grid = np.zeros(len(x_distrib) + 1)
                x_gr = np.sqrt(x_distrib[1:] * x_distrib[:-1])

                x_grid[0] = x_gr[0] / ratio
                x_grid[-1] = x_gr[-1] * ratio

                x_grid[1:-1] = x_gr

                self.dx = np.diff(x_grid)

            # Distribution
            distrib = np.asarray(distrib)

            self.num_distrib = len(distrib)
            self.distribProf = None

            if mass > 0:
                distrib = distrib / distrib.sum()
                if distrib_type == 'vol_perc':
                    distr = (mass / dens) * distrib / kv / x_distrib**3 \
                        * 1e18 / self.dx   # number/um

                elif distrib_type == 'mass_perc':
                    distr = mass*distrib / x_distrib**3 / kv * 1e18

            else:
                mom_three = self.getMoments(mom_num=3)  # m**3
                mass = mom_three * self.kv * dens
                distr = distrib

            distr[distr == 0] = eps
            self.distrib = distr

            if self.num_mom is None:
                self.moments = self.getMoments()
            else:
                idx = np.arange(self.num_mom)
                self.moments = self.getMoments(mom_num=idx)

        # Mass
        self.mass = mass
        self.vol = mass / dens  # m**3

        mw_av = np.dot(self.mole_frac, self.mw)
        self.moles = mass / mw_av

        # # Moles
        # self.moles = moles
        # self.mole_frac = mole_frac

        if mass_frac is not None:
            sum_fracs = sum(mass_frac)
            if sum_fracs < 0.99:
                raise RuntimeError('The sum of mass fractions is less than 0.99')

        self.moisture = moisture

        self.porosity = porosity

    def updatePhase(self, distrib=None, mass=None):
        if distrib is not None:
            self.distrib = distrib
            self.moments = self.getMoments(distrib)

        if mass is not None:
            self.mass = mass
            self.vol = mass / self.getDensity()

    def getMoments(self, distrib=None, mom_num=None):
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
                integrand = distrib * self.x_distrib**exp
                moments[ind] = trapezoidal_rule(self.x_distrib, integrand.T)

            if len(mom_ind) == 1:
                moments = moments[0]

        else:
            moments = np.zeros((len(distrib), len(mom_ind)))
            for ind, exp in enumerate(mom_ind):
                integrand = distrib * self.x_distrib**exp
                moments[:, ind] = trapezoidal_rule(self.x_distrib, integrand.T)

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


    def getPorosity(self, distrib=None, diam_filter=1, AR=None, sphericity=None):  # x_distrib is the x

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
        node_x_dist = (x_dist[:-1] + x_dist[1:])/ 2
        node_CSD = (distrib[:-1] + distrib[1:])/ 2

        vol_cry = node_CSD * del_x_dist * (kv * node_x_dist**3) # Volume of crystals in each bin
        frac_vol_cry = vol_cry/ np.sum(vol_cry)

        vol_particle = kv * node_x_dist**3
        d_part_sphere = (6 * vol_particle/ np.pi)**(1/3)
        d_part_equiv_pack = d_part_sphere/ (sphericity**2.785 *
                                            np.exp(2.946 * (1 - sphericity)))

        #Initial porosity
        D_mean = mom_one/ mom_zero
        E_0_Jeschar = 0.375 + 0.34 * D_mean/ diam_filter # # average porosity of packing of uniform sized spheres [-]

        initial_porosity = E_0_Jeschar

        V = 1/ (1 - initial_porosity) * np.ones_like(node_x_dist)   #Specific Volume for initial porosity

        # Evaluate specific volume using modified linear packing model
        V_T_node = np.zeros(len(node_x_dist))

        for i in range(len(node_x_dist)):

            sum_V_large_term= 0
            sum_V_small_term = 0

            for j in range(i):
                r = d_part_equiv_pack[j]/ d_part_equiv_pack[i]
                g_r = (1 - r)**2 + 0.4*r*(1-r)**3.7
                V_large_j = V[j] - (V[j] - 1) * g_r - V[i]
                sum_V_large_term += V_large_j * frac_vol_cry[j]

            for j in range(i+1, len(node_x_dist)):
                r = d_part_equiv_pack[i]/ d_part_equiv_pack[j]
                f_r = (1 -r)**3.3 + 2.8*r*(1 - r)**2.7
                V_small_j = V[j] * (1 - f_r) - V[i]
                sum_V_small_term += V_small_j * frac_vol_cry[j]

            #V_T_node[i] = vol_cry[i] * frac_vol_cry[i] + sum_V_large_term + sum_V_small_term
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
