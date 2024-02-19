#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:13:10 2020

@author: dcasasor
"""

import numpy as np
from PharmaPy.Gaussians import gaussian


def getBipartiteNames(first, second):
    """
    Create bipartite graph for matching phases. It only matches phase names.
    It should rely on some thermodynamic criterion to decide compatible phases.

    Parameters
    ----------
    first : dict
        name dict of the upstream UO.
    second : dict
        name dict of the downstream UO.

    Returns
    -------
    graph : dict
        bipartite graph.

    """
    graph = {}
    for two in second:
        for one in first:
            if one == second:
                graph[two] = one
                break

    return graph


def getBipartiteMultiPhase(first, second):
    """
    Creates bipartite graph for input dictionaries specifying phases and states

    Parameters
    ----------
    first : dict
        DESCRIPTION.
    second : dict
        DESCRIPTION.

    Returns
    -------
    graph : dict

    """

    upper_graph = getBipartiteNames(first, second)

    graph = {}
    for phase_down, phase_up in upper_graph.items():
        key = '/'.join(phase_down, phase_up)
        graph[key] = getBipartite(first[phase_up], second[phase_down])

    return graph


def getBipartite(first, second):
    graph = {}
    types = {}
    comp_names = ['conc', 'frac']
    amount_names = ['mass', 'moles', 'vol']

    num_first = len(first)
    for two in second:
        for count, one in enumerate(first):
            if 'distr' in one and 'solid_conc' in two:
                graph[two] = one
                types['distrib'] = (one, two)
                break
            elif any(word in one for word in comp_names) and any(word in two for word in comp_names):
                graph[two] = one
                types['composition'] = (one, two)
                break
            elif 'flow' in one and 'flow' in two:
                graph[two] = one
                types['flow'] = (one, two)
                break
            elif 'distr' in one and 'distr' in two:
                graph[two] = one
                types['distrib'] = (one, two)
                break
            elif any(one == word for word in amount_names) and any(two == word for word in amount_names):
                graph[two] = one
                types['amount'] = (one, two)
            elif one == two:
                graph[two] = one
                break
            elif count == num_first - 1:
                graph[two] = None

    return graph, types


def get_types(names):
    conc_type = [name for name in names
                 if ('conc' in name) or ('frac' in name)][0]

    distrib_type = [name if 'distr' in name else None
                    for name in names][0]

    distrib_type = list(filter(lambda x: 'distr' in x, names))

    flow_type = list(filter(lambda x: 'flow' in x, names))

    amount_type = list(
        filter(lambda x: x == 'mass' or x == 'moles' or x == 'vol', names))

    if not distrib_type:
        distrib_type = None
    else:
        distrib_type = distrib_type[0]

    if not flow_type:
        flow_type = None
    else:
        flow_type = flow_type[0]

    if not amount_type:
        amount_type = None
    else:
        amount_type = amount_type[0]

    return conc_type, distrib_type, flow_type, amount_type


def get_dict_states(names, num_species, num_distr, states):
    count = 0
    dict_states = {}

    for name in names:
        if 'conc' in name or 'frac' in name:
            idx_composition = range(count, count + num_species)
            dict_states[name] = states.T[idx_composition].T

            count += num_species
        elif 'distrib' in name or 'mu_n' in name:
            idx_distrib = range(count, count + num_distr)
            dict_states[name] = states.T[idx_distrib].T

            count += num_distr
        else:
            dict_states[name] = states.T[count].T

            count += 1

    return dict_states


class NameAnalyzer:
    def __init__(self, names_up, names_down, num_species, num_distr=None):
        self.names_up = names_up
        self.names_down = names_down

        self.num_species = num_species
        self.num_distr = num_distr

        self.bipartite, self.conv_types = getBipartite(names_up, names_down)

    def get_idx(self):
        count = 0

        idx_flow = None
        idx_amount = None
        idx_distrib = None

        for name in self.names_up:
            if 'conc' in name or 'frac' in name:
                idx_composition = range(count, count + self.num_species)

                count += self.num_species
            elif 'distrib' in name:
                idx_distrib = range(count, count + self.num_distr)

                count += self.num_distr
            else:
                if 'flow' in name:
                    idx_flow = count
                elif name == 'mass' or name == 'moles' or name == 'vol':
                    idx_amount = count

                count += 1

        return idx_composition, idx_flow, idx_amount, idx_distrib

    def convertUnits(self, matter_transf):
        # if matter_transf.__module__ == 'PharmaPy.MixedPhases':  # TODO: not general
        #     matter_transf = matter_transf.Liquid_1

        dict_in = matter_transf.y_upstream

        dict_out = {}

        for target, source in self.bipartite.items():
            if source is not None:

                if target != source:
                    y_j = dict_in[source]
                    
                    if 'distrib' in target or 'solid_conc' in target:
                        converted_state = self.__convert_distrib(
                            source, target, y_j, matter_transf)

                    elif 'conc' in target or 'frac' in target:
                        converted_state = self.__convertComposition(
                            source, target, y_j, matter_transf)

                    elif 'flow' in target:
                        comp = self.conv_types['composition']
                        
                        converted_state = self.__convertFlow(
                            source, target, y_j, matter_transf,
                            dict_in[comp[0]], comp[0])

                    dict_out[target] = converted_state

                else:
                    dict_out[target] = dict_in[source]

        return dict_out

    def __convertComposition(self, prefix_up, prefix_down, composition,
                             matter_object):
        up, down = prefix_up, prefix_down

        if 'frac' in up and 'frac' in down:
            method_name = 'frac_to_frac'
            if 'mole' in up:
                fun_kwargs = {'mole_frac': composition}
            elif 'mass' in up:
                fun_kwargs = {'mass_frac': composition}

        elif 'frac' in up and 'conc' in down:
            method_name = 'frac_to_conc'

            if 'mole' in up:
                fun_kwargs = {'mole_frac': composition}
            elif 'mass' in up:
                fun_kwargs = {'mass_frac': composition}

            if 'mass' in down:
                fun_kwargs['basis'] = 'mass'

        elif 'mole_conc' in up and 'frac' in down:
            method_name = 'conc_to_frac'
            fun_kwargs = {'conc': composition}

            if 'mass' in down:
                fun_kwargs['basis'] = 'mass'
            else:
                fun_kwargs['basis'] = 'mole'

        elif 'mass_conc' in up and 'frac' in down:
            method_name = 'mass_conc_to_frac'
            fun_kwargs = {'conc': composition}

            if 'mole' in down:
                fun_kwargs['basis'] = 'mole'

        elif 'conc' in up and 'conc' in down:
            method_name = 'conc_to_conc'

            if 'mole' in up:
                fun_kwargs = {'mole_conc': composition}
            else:
                fun_kwargs = {'mass_conc': composition}

        method = getattr(matter_object, method_name, None)
        if method is None:
            for phase in matter_object.Phases:
                if hasattr(phase, method_name):
                    method = getattr(phase, method_name)

                    break

        output_composition = method(**fun_kwargs)

        return output_composition

    def __convertFlow(self, prefix_up, prefix_down, flow, matter_object,
                      composition, comp_name):
        up, down = prefix_up, prefix_down

        # Molecular weight
        if np.asarray(flow).ndim == 0:
            mw_av = matter_object.mw_av
        elif comp_name == 'mole_frac':
            mw_av = np.dot(matter_object.mw, composition.T)
        elif comp_name == 'mass_frac':
            mole_frac = matter_object.frac_to_frac(mass_frac=composition)
            mw_av = np.dot(matter_object.mw, mole_frac.T)
        elif comp_name == 'mole_conc':
            mole_frac = (composition.T / composition.sum(axis=1))
            mw_av = np.dot(matter_object.mw, mole_frac)

        # Density
        if np.asarray(flow).ndim == 0:
            density = matter_object.getDensity()
        elif comp_name == 'mole_frac' or comp_name == 'mass_frac':
            density = matter_object.getDensity(**{comp_name: composition})
        elif comp_name == 'mole_conc':
            density = matter_object.getDensity(mole_frac=mole_frac.T)

        # Convert units
        if 'mass' in up and 'mole' in down:
            flow_out = flow / mw_av * 1000  # mol/s
        elif 'mole' in up and 'mass' in down:
            flow_out = flow * mw_av / 1000  # kg/s
        elif 'vol' in down:
            if 'mole' in up:
                density *= 1000 / mw_av

            flow_out = flow / density  # m3/s

        elif 'vol' in up:
            dens = matter_object.getDensity()  # kg/m3
            if 'mole' in down:
                dens *= 1000 / mw_av

            flow_out = flow * dens  # kg/s - mol/s

        return flow_out

    def __convert_distrib(self, prefix_up, prefix_down, distrib,
                          matter_object):
        up, down = prefix_up, prefix_down

        if 'distrib' in up and 'total' in down:
            out = distrib
        elif 'num' in up and 'vol' in down:
            pass
        elif up == 'distrib' and down == 'solid_conc':
            out = matter_object.getSolidsConcentr(distrib=distrib,
                                                  basis='mass')

        return out


if __name__ == '__main__':
    names_up = ['temp', 'mole_conc', 'vol_flow']
    names_down = ['mole_frac', 'temp', 'num_distrib', 'mass_flow']

    num_species = 2
    num_distr = 100

    # Data
    states_up = np.array([300, 0.7, 1, 0.001])
    states_down = np.array([0.35, 0.65, 320])

    x_distrib = np.linspace(0, 1000, num_distr)
    distrib = gaussian(x_distrib, 400, 10, 1e10)

    states_down = np.append(states_down, distrib)
    states_down = np.append(states_down, 1)

    # With name analyzer
    analyzer = NameAnalyzer(names_up, names_down, num_species,
                            num_distr)
