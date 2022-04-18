# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:36:35 2020

@author: dcasasor
"""

from PharmaPy.NameAnalysis import NameAnalyzer, get_dict_states

import numpy as np
import copy


def get_inputs(time, uo, num_species, num_distr=0):
    Inlet = getattr(uo, 'Inlet', None)

    names_upstream = uo.names_upstream
    names_states_in = uo.names_states_in
    bipartite = uo.bipartite
    if Inlet is None:
        input_dict = {}

        return input_dict

    elif Inlet.y_upstream is None or len(Inlet.y_upstream) == 1:
        # this internally calls the DynamicInput object if not None
        input_dict = Inlet.evaluate_inputs(time)

        for name in names_states_in:
            if name not in input_dict.keys():
                val = getattr(Inlet, name, None)
                if val is None:  # search in subphases inside Inlet
                # if hasattr(uo, 'states_in_phaseid'):
                    obj_id = uo.states_in_phaseid[name]
                    instance = getattr(Inlet, obj_id)
                    val = getattr(instance, name)

                input_dict[name] = val

    else:
        all_inputs = Inlet.InterpolateInputs(time)
        input_upstream = get_dict_states(names_upstream, num_species,
                                         num_distr, all_inputs)

        input_dict = {}
        for key in names_states_in:
            val = input_upstream.get(bipartite[key])
            if val is None:
                val = uo.input_defaults[key]

            input_dict[key] = val

    return input_dict


class Graph:
    def __init__(self, connections=None):

        self.connections = connections

        # Create graph
        self.graph = {}
        self.get_graph()

        self.num_vert = len(self.graph)

    def get_graph(self):
        for conn in self.connections:
            if conn.source_uo in self.graph.keys():
                self.graph[conn.source_uo].append(conn.destination_uo)
            else:
                self.graph[conn.source_uo] = [conn.destination_uo]

            if conn.destination_uo not in self.graph.keys():
                self.graph[conn.destination_uo] = []

        self.vertices = list(self.graph.keys())

        # for name in self.vertices:
        #     if 'Source' == name.__class__.__name__:
        #         self.graph.pop(name)

    # A recursive function used by topologicalSort
    def __topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if not visited[i]:
                # TODO this is new for me!!
                self.__topologicalSortUtil(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = {key: False for key in self.vertices}
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for vert in self.vertices:
            if not visited[vert]:
                self.__topologicalSortUtil(vert, visited, stack)

        # Print contents of stack
        return stack


class Connection:
    def __init__(self, material=None, source_uo=None, destination_uo=None,
                 mass_flow=0, mass_ufunc=None, mole_flow=0,
                 vol_flow=0, vol_ufunc=None):

        if source_uo is None and material is None:
            raise RuntimeError("No source UO to take material from. Please "
                               "specify entering material with the 'material' "
                               "argument")
        self.Matter = material

        self.source_uo = source_uo
        self.destination_uo = destination_uo
        self.y_list = []

    def ReceiveData(self):
        # Source UO
        if self.Matter is None:
            self.Matter = self.source_uo.Outlet
            self.num_species = self.Matter.num_species

            self.Matter.y_upstream = self.source_uo.outputs

            if self.source_uo.is_continuous:
                self.Matter.time_upstream = self.source_uo.timeProf
            else:
                self.Matter.time_upstream = self.source_uo.timeProf[-1]

            self.y_list.append(self.source_uo.outputs)

        else:
            if self.source_uo.is_continuous:
                self.y_list.append(self.source_uo.outputs[1:])

                y_stacked = np.vstack(self.y_list)
                self.y_list = [y_stacked]

                time_prof = self.source_uo.timeProf
                idxs = np.where(np.diff(time_prof) > 0)[0]

                time_prof = np.concatenate([time_prof[idxs],
                                           np.array([time_prof[-1]])])

                self.Matter.y_upstream = y_stacked
                self.Matter.time_upstream = time_prof

    def TransferData(self):
        if self.source_uo is None:
            states_up = None
        else:
            states_up = self.source_uo.names_states_out

        if self.destination_uo.__class__.__name__ == 'DynamicCollector':
            if self.source_uo.__class__.__name__ == 'MSMPR':
                states_down = self.destination_uo.names_states_in[1]
                self.destination_uo.name_idx = 1
            else:
                states_down = self.destination_uo.names_states_in[0]

        else:
            states_down = self.destination_uo.names_states_in

        if states_up is None:
            bipartite = None
            names_upstream = None
        elif self.source_uo.oper_mode == 'Batch':
            bipartite = None
            names_upstream = None
        else:
            name_analyzer = NameAnalyzer(states_up, states_down,
                                         self.num_species,
                                         0)

            # Convert units and pass states to self.Matter
            name_analyzer.convertUnits(self.Matter)

            bipartite = name_analyzer.bipartite
            names_upstream = name_analyzer.names_up

        if self.destination_uo.__class__.__name__ == 'Mixer':
            self.destination_uo.bipartite.append(bipartite)
            self.destination_uo.names_upstream.append(names_upstream)
        else:
            self.destination_uo.bipartite = bipartite
            self.destination_uo.names_upstream = names_upstream

        # ---------- Destination UO
        mode = self.destination_uo.oper_mode
        transfered_matter = copy.deepcopy(self.Matter)

        if mode == 'Batch':
            self.destination_uo.Phases = transfered_matter
            self.destination_uo.material_from_upstream = True
        elif mode == 'Semibatch':
            if self.destination_uo.Phases is None:
                self.destination_uo.Phases = transfered_matter
                self.destination_uo.material_from_upstream = True

        else:  # Continuous
            source_phases = self.source_uo.Outlet
            if self.source_uo.oper_mode == 'Batch' and source_phases is not self.Matter:
                if hasattr(self.Matter, 'Phases') and \
                        hasattr(source_phases, 'Phases'):
                    pass
                elif hasattr(self.Matter, 'Phases'):
                    pass

                elif hasattr(source_phases, 'Phases'):
                    print('Warning: Source UO yields a '
                          'MixedPhases object, whereas the destination stream '
                          'is a %s object' % self.Matter.__class__.__name__)

                    name_destin = transfered_matter.__class__.__name__
                    for phase in source_phases.Phases:
                        name_source = phase.__class__.__name__

                        if 'Liquid' in name_source and 'Liquid' in name_destin:
                            transfered_matter.updatePhase(
                                mass_frac=phase.mass_frac)
                            transfered_matter.temp = phase.temp

                        elif 'Solid' in name_source and 'Solid' in name_destin:
                            pass

            if hasattr(self.destination_uo, 'Inlet'):
                self.destination_uo.Inlet = transfered_matter
                self.destination_uo.material_from_upstream = True
            else:
                self.destination_uo.Inlets = transfered_matter
                self.destination_uo.material_from_upstream = True

            if self.destination_uo.__class__.__name__ == 'DynamicCollector':
                if self.source_uo.__module__ == 'PharmaPy.Crystallizers':
                    self.destination_uo.KinCryst = self.source_uo.Kinetics
                    self.destination_uo.kwargs_cryst = {
                        'target_ind': self.source_uo.target_ind,
                        'target_comp': self.source_uo.target_comp,
                        'scale': self.source_uo.scale}
