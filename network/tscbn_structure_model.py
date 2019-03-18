#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import random

from network.base_structure_model import BaseStructureModel
from network.tscbn import TSCBN
from _include.m_libpgm.graphskeleton import GraphSkeleton

import numpy as np
import copy


class TSCBNStructureModel(BaseStructureModel):
    '''
    This generator creates a Temporal state change BN from the given specification
    '''

    def __init__(self):
        super(TSCBNStructureModel, self).__init__()
        self._diabled_previous_inter_edge = True
        self._temporal_variance = 0.01

    def generate_model(self, structure_specification):

        # 1. create defined number of nodes per object
        v, node_cpds, temp_dict = self._create_nodes(structure_specification)

        # 2. create edges incl. time between vertices
        e, temp_gap_dict = self._create_edges(v, structure_specification, temp_dict)

        # 3. add temporal information
        inverted_temp_dict = self._invert_dict(temp_dict)
        self._temporal_information(v, temp_gap_dict, node_cpds, self._parents_dict_from_edges(e), temp_dict,
                                   inverted_temp_dict)  # node_cpds passed by reference and contain temporal information

        # 4. Skeleton
        skel = GraphSkeleton()
        skel.V = v
        skel.E = e
        skel.toporder()

        # 5. Create Model
        tbn = TSCBN("", skel, node_cpds, unempty=True, forbid_never=True,
                    discrete_only=True)  # Discrete case - later continuous nodes

        # 6. Set cpds of value nodes
        self._set_cpds(tbn, structure_specification)

        return tbn

    def _set_cpds(self, tbn, structure_specification):

        # 1. Probability that a state change occurred
        prob_state_change = structure_specification["state_change"]

        # 2. Node name defined
        for object_idx in range(len(prob_state_change)):
            for node_idx in range(len(prob_state_change[object_idx])):

                # get relevant data
                if node_idx == 0: continue
                vertex = "V%s_%s" % (str(object_idx), str(node_idx))
                parent_vertex = "V%s_%s" % (str(object_idx), str(node_idx - 1))
                probability_no_sc = 1.0 - prob_state_change[object_idx][node_idx]

                # set diagonals to zero
                if isinstance(tbn.nodes[vertex].Vdataentry["cprob"], dict):
                    for k in tbn.nodes[vertex].Vdataentry["cprob"]:
                        # set all diagonals to zero - get preceding parent
                        parent_idx = tbn.nodes[vertex].Vdataentry["parents"].index(parent_vertex)
                        obj = eval(k)[parent_idx]
                        val_idx = tbn.nodes[vertex].Vdataentry["vals"].index(obj)

                        #
                        tbn.nodes[vertex].Vdataentry["cprob"][k][val_idx] = 0.0
                        tbn.nodes[vertex].Vdataentry["cprob"][k] = (1.0 - probability_no_sc) * (
                                    tbn.nodes[vertex].Vdataentry["cprob"][k] / sum(
                                tbn.nodes[vertex].Vdataentry["cprob"][k]))
                        tbn.nodes[vertex].Vdataentry["cprob"][k][val_idx] = probability_no_sc

                        tbn.Vdata[vertex]["cprob"][k] = tbn.nodes[vertex].Vdataentry["cprob"][k]

    def _invert_dict(self, d_dict):
        out_d = {}

        for d in d_dict:
            out_d[d_dict[d]] = d
        return out_d

    def _parents_dict_from_edges(self, edges):
        '''
        From a list of edges extract parents per edge
        '''
        res_dict = {}
        for e in edges:
            if e[1] in res_dict and not e[0] in res_dict[e[1]]:
                res_dict[e[1]].append(e[0])
            else:
                res_dict[e[1]] = [e[0]]
        return res_dict

    def _temporal_information(self, vertices, temp_gap_dict, node_cpds, parents, temp_dict, inv_temp_dict):
        # Mean value ist der Abstand wenn ich alle meine Parents anschaue und dann
        # den nehme der am nähesten bei mir ist
        for v in vertices:
            # welches ist spaeter passiert
            # - has no parent
            variance = self._temporal_variance
            if v not in parents:
                dL = 0
            else:
                # has parent
                pars = np.array([k for k in temp_dict if temp_dict[k] in parents[v]])
                if len(pars) == 0:
                    fr_n = parents[v][0]
                else:
                    idx = max(pars)
                    fr_n = temp_dict[idx]
                # keey = str([fr_n, v])
                try:
                    dL = inv_temp_dict[v] - inv_temp_dict[fr_n]  # temp_gap_dict[keey]
                except:
                    dL = inv_temp_dict[v]  # if it does not appear it is a parent
            if v.split("_")[-1] == "0":
                dL = 0
                variance = 0
            node_cpds[v]["dL_mean"] = dL
            node_cpds[v]["dL_var"] = variance  # assume little variance

    def _create_nodes(self, spec):
        v, node_cpds = [], dict()
        temp_dict = {}  # key: absolute time, value: vertex

        # Initial nodes
        for i in range(spec["object_number"]):
            number_of_nodes_per_obj = spec["per_object_chain_number"][i]
            obj_name = spec["object_names"][i]
            node_names = self._get_node_names(obj_name, number_of_nodes_per_obj)
            states = spec["object_states"][obj_name]
            # t_gaps = spec["temp_gap_between_objects"]
            t = 0
            kk = -1
            for n_name in node_names:
                kk += 1
                v += self._dynamic_node(n_name, "disc", states,
                                        node_cpds)  # self._dynamic_node(n_name, "disc", states + ["Never"], node_cpds)
                temp_dict[t] = n_name
                try:
                    t += spec["temp_gap_between_objects"][i][kk]
                except:
                    pass
        return v, node_cpds, temp_dict

    def _create_edges(self, vertices, structure_specification, temp_dict):
        e = []
        self._temporal_variance = structure_specification["temporal_variance"]

        # 1. add self dependencies
        e += self._self_dependencies(vertices, [u.replace("O", "V") for u in structure_specification["object_names"]])

        # 2. add connection between objects - Abh. von mir und meinem Vorgaenger
        # kenne pro Objekt die Distanz zwischen den Zustandswechseln
        # store all abs times to values - then choose what I need
        temp_gap = {}  # stores the gap between object and its son per object
        times_a = []
        for i in range(structure_specification["object_number"]):
            # get relevant subset of our dictionary
            rel_sub_list = structure_specification['inter_edges_to_this_object'][i] + [
                structure_specification["object_names"][i]]
            sub_dict = self._get_subdict(rel_sub_list, temp_dict)
            target = structure_specification["object_names"][i].replace("O", "V")

            latest_time_of_target = sorted([k for k in temp_dict.keys() if str.startswith(temp_dict[k], target)])[-1]
            # choose number of destination nodes to connect
            try:
                sub_dict = self._choose_subset_of_nodes(structure_specification, sub_dict, i, temp_dict,
                                                        latest_time_of_target)
            except:
                sub_dict = self._choose_subset_of_nodes(structure_specification, sub_dict, i, temp_dict,
                                                        latest_time_of_target)

            times = sorted(list(sub_dict.keys()))
            times_a += [times]

            last = {}
            for time in times:
                if temp_dict[time].split("_")[0] == target:
                    # edge from all entries in last to target
                    for app_time in last:
                        new_e = [last[app_time], temp_dict[time]]
                        # Zusaetzlich brauche Verbindung zum Vorgaenger per Definition (oder auch nicht)
                        if not self._diabled_previous_inter_edge:
                            nex = int(last[app_time].split("_")[1]) - 1
                            if nex != -1:
                                fr_n = last[app_time].split("_")[0] + "_" + str(nex)
                                new_e2 = [fr_n, temp_dict[time]]

                                if not new_e2 in e:
                                    e += [new_e2]
                                    t = None
                                    for k, v in temp_dict.items():
                                        if v == fr_n:
                                            t = k
                                            break
                                    # store temporal gap to this object
                                    if str(nex) == "0": k = 0.0
                                    temp_gap[str(new_e2)] = time - k

                        # Add if not existing
                        if not new_e in e:
                            e += [new_e]
                            # store temporal gap to this object
                            temp_gap[str(new_e)] = time - app_time

                    last = {}
                else:
                    last[time] = temp_dict[time]

        return e, temp_gap

    def _random_pick_uniform(self, elements, nr_elements):
        res = []
        l = copy.deepcopy(list(elements.keys()))
        for _ in range(nr_elements):
            idx = l[round(random.random() * len(l)) - 1]
            res += [elements[idx]]
            l.remove(idx)
        return res

    def _choose_subset_of_nodes(self, structure_specification, sub_dict, i, temp_dict, latest_time_of_target):
        '''
        Fordere das mein letzter Knoten der letze sein muss!
        :param structure_specification:
        :param sub_dict:
        :param i:
        :param temp_dict:
        :return:
        '''
        node_number_per_object = int(structure_specification['nodes_per_object'][i])
        pot_connects = self._get_subdict(structure_specification['inter_edges_to_this_object'][i], temp_dict)

        # pot connects are only the ones occurring before me!
        if len(pot_connects) < node_number_per_object:
            raise ValueError(
                "Number of nodes to connect is %s, while only %s valid nodes were found - change percentage_inter value (lower)" % (
                str(node_number_per_object), str(pot_connects)))
        pot_connects = dict([(p, pot_connects[p]) for p in pot_connects if p < latest_time_of_target])

        nodes_to_connect = self._random_pick_uniform(pot_connects, node_number_per_object)
        rel_dict = {}
        dest_vals = list(
            self._get_subdict([structure_specification["object_names"][i]], temp_dict).values()) + nodes_to_connect
        for k in sub_dict:
            if sub_dict[k] in dest_vals:
                rel_dict[k] = sub_dict[k]
        return rel_dict

    def _get_subdict(self, rel_objects, temp_dict):
        '''
        takes a list of relevant objects and returns the relevant part of the dictionary
        '''
        rel_vers = [u.replace("O", "V") for u in rel_objects]
        sub = {}
        for k in temp_dict:
            if temp_dict[k].split("_")[0] in rel_vers:
                sub[k] = temp_dict[k]
        return sub

    def _get_node_names(self, obj_name, number_of_nodes_per_obj):
        new_name = obj_name.replace("O", "V")
        return ["%s_%s" % (str(new_name), str(k)) for k in range(number_of_nodes_per_obj)]

    def unique(self, sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    def _dynamic_node(self, node_name, type_dc, vals, node_cpds):
        node_cpds[node_name] = dict()
        node_cpds[node_name]["vals"] = self.unique(
            [str(a) for a in vals if not str(a) == "nan"])  # nan is not an outcome!
        node_cpds[node_name]["type_dc"] = type_dc  # preliminary - will be adjusted depending on parents!
        return [node_name]

    def _self_dependencies(self, vertices, rel_nodes):
        try:
            edges = []
            verts = sorted(vertices)
            fr = verts[0]

            for v in verts[1:]:
                to = v
                fst = fr.split("_")[0]  # ''.join([i for i in fr if not i.isdigit()])
                if fst == to.split("_")[0] and fst in rel_nodes:
                    edges.append([fr, to])
                fr = to
            return edges
        except:
            return []

    def _self_dependencies_digit_format(self, vertices, rel_nodes):
        try:
            edges = []
            verts = sorted(vertices)
            fr = verts[0]

            for v in verts[1:]:
                to = v
                fst = ''.join([i for i in fr if not i.isdigit()])
                if fst == ''.join([i for i in to if not i.isdigit()]) and fst in rel_nodes:
                    edges.append([fr, to])
                fr = to
            return edges
        except:
            return []

    def _print_tree_info(self, nodes):
        print("\n ------------------ \n Tree Information \n ------------------ ")
        a_ll = copy.deepcopy(list(nodes.keys()))

        for obj_id in range(len(a_ll)):
            fst = True
            for state_id in range(len(a_ll)):
                try:
                    ver = "V%s_%s" % (str(obj_id), str(state_id))
                    gap_ver = "dL_%s" % ver
                    nodes[ver]
                    if fst:
                        print("\nCurrent Object: %s" % str(obj_id))
                        fst = False
                    print(ver + " Parents: " + str(nodes[ver].Vdataentry["parents"]))
                    print(gap_ver + " Parents: " + str(nodes[gap_ver].Vdataentry["parents"]))
                    print("Mean dL: %s" % str(nodes[gap_ver].Vdataentry["mean_base"]))
                    print("\n")
                except:
                    break
