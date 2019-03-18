import itertools
import numpy as np
import random
from network.tscbn_structure_model import TSCBNStructureModel
from copy import deepcopy

class ModelGenerator(object):
    '''
        Generates various structures as specified

        The Goal is to network objects that are in a state for a certain amount of time and
        that change their state and depend on the state of other objects
        With this generator a test-case for such a scenario is created
        '''

    def __init__(self):
        self._generator_model = TSCBNStructureModel()

        self._sc_probability = [0.8, 0.8]
        self._intra_object_temp_range = [0.5, 1.0]
        self._temporal_variance = 0.001
        self._dbn_tolerance = 0.1

        self.show_plot_generated = False  # show generated models
        self.show_text_generated = False

    def new_tscbn(self):
        models, specifications = self._run_spec_structure()
        tscbn = models[self._generator_model.model_key()]
        self.set_CPDs(tscbn)
        return tscbn

    # def run_next_testcase(self):
    #     '''
    #     Returns the next test case
    #     '''
    #     # Draw structure
    #     return self._run_spec_structure()

    def _run_spec_structure(self):
        '''
        Generate testcase as specified by a specification that is set in the
        generator
        :return:
        '''
        specification = self._draw_spec_structure()

        # 2. generate network for this setting
        models = {}
        models[self._generator_model.model_key()] = self._generator_model.generate_model(specification)
        models[self._generator_model.model_key()].show_plot_generated = self.show_plot_generated
        models[self._generator_model.model_key()].show_text_generated = self.show_text_generated

        # 3. return all generated models
        return models, specification

    def _draw_uniform(self, in_range):
        if in_range[0] == in_range[1]: return in_range[0]
        return in_range[0] + random.random() * (in_range[-1] - in_range[0])

    def _draw_uniform_float(self, min_max):
        min_val, max_val = min_max
        return min_val + (max_val - min_val) * random.random()

    def _draw_spec_structure(self):
        result = {}

        # Set temporal variance per node - ONLY FOR TSCBN!
        result["temporal_variance"] = self._temporal_variance

        # Set tolerance percentage in DBN - a slice is not allowed to be further away than this
        result["dbn_tolerance"] = self._dbn_tolerance

        # Draw number of objects to create
        object_number = round(self._draw_uniform(self._object_range))
        result["object_number"] = object_number

        # Draw number of nodes per object
        per_object_chain_number = [round(self._draw_uniform(self._temp_node_range)) for _ in range(object_number)]
        result["per_object_chain_number"] = per_object_chain_number

        # Draw number of states per node
        states_per_object = [round(self._draw_uniform_float(self._state_range)) for _ in range(object_number)]
        result["states_per_object"] = states_per_object

        # Draw probability of a state change per node
        result["state_change"] = [[self._draw_uniform_float(self._sc_probability)
                                   for _ in range(per_object_chain_number[i])] for i in range(object_number)]

        # Draw temporal gap between intra-nodes
        temp_gap_between_objects = [[self._draw_uniform_float(self._intra_object_temp_range)
                                     for _ in range(per_object_chain_number[i] - 1)] for i in range(object_number)]
        result["temp_gap_between_objects"] = temp_gap_between_objects

        # Set object and state names
        result["object_names"], result["object_states"] = self._set_object_properties(object_number, states_per_object)

        # Draw number of objects that connect to this object
        result["inter_edges_to_this_object"] = self._draw_objects_to_connect(object_number, result["object_names"])

        # Draw number of nodes per object
        result["nodes_per_object"] = [round(kk * self._draw_uniform_float(self._percentage_inter_edges)) for kk in
                                      per_object_chain_number]

        return result

    def set_node_range(self, min_objects, max_objects, min_temp_nodes, max_temp_nodes, min_states, max_states):
        '''
        This method sets the parameters for the node creation. Each object has several states that change over time.
        Per test case different numbers of objects and states within specified ranges are created. The temporal
        length of the chain to be created is defined by min_temp_nodes and max_temp_nodes, which is the range
        within which each object has nodes. E.g. object 1 could have 3 nodes
        '''
        self._object_range = [min_objects, max_objects]
        self._temp_node_range = [min_temp_nodes, max_temp_nodes]
        self._state_range = [min_states, max_states]

    def set_state_change_probability(self, min_probability, max_probability):
        '''
        Defines probability within which the probability that a state change happens lies
        If 1.0 state change will always occur at 0.0 no state will change
        '''
        self._sc_probability = [min_probability, max_probability]

    def set_temporal_range(self, min_per_object_gap, max_per_object_gap):
        '''
        Defines the temporal range for an object - with which one value occurs
        after the other - also specify the average distance within one object occurs after
        another
        '''
        self._intra_object_temp_range = [min_per_object_gap, max_per_object_gap]
        # range zwischen Objekten ist durch restliche Parameter definierts

    def set_connection_ranges(self, min_edges_per_object, max_edges_per_object,
                              min_percent_inter=0.0, max_percent_inter=1.0):
        '''
        Defines the number of edges between
        similarity_variation: Defines the deviation per inter node connection that is possible
                              0 means no variation. I.e. the number of specified edges is always the same
                              e.g. 1 means - one edge could be missing/added between nodes
        edges_per_node        Defines the number of inter node/object connections that are possible
        '''
        self._edges_inter_object_range = [min_edges_per_object, max_edges_per_object]
        self._percentage_inter_edges = [min_percent_inter, max_percent_inter]

    def _set_object_properties(self, object_number, states_per_object):
        object_names, object_states = ["O%s" % str(i) for i in range(object_number)], {}
        for i in range(len(object_names)):
            object_states[object_names[i]] = ["o%s_%s" % (str(i), str(j)) for j in range(states_per_object[i])]
        return object_names, object_states

    def _draw_objects_to_connect(self, object_number, object_names):
        inter_edges_to_this_object_pre = [self._draw_uniform(self._edges_inter_object_range) for _ in
                                          range(object_number)]
        if np.any(np.array(inter_edges_to_this_object_pre) >= object_number): raise AssertionError(
            "Number of connecting edges needs to be smaller then object number")

        # per edge draw from this range and remove an edge in this range
        t = -1
        inter_edges_to_this_object = []
        for obj_edge_num in inter_edges_to_this_object_pre:
            t += 1
            p_list = deepcopy(object_names)
            p_list.remove(object_names[t])  # edge to myself is meaningless for inter edges - remove it
            object_edges = self._draw_uniform_samples_from_list(p_list, int(obj_edge_num))
            inter_edges_to_this_object.append(object_edges)

        return inter_edges_to_this_object

    def _draw_uniform_samples_from_list(self, lst, sample_nr):
        ''' Draw sample_nr random samples from a list'''
        res = []
        for _ in range(sample_nr):
            idx = round(self._draw_uniform([0, len(lst) - 1])) - 1
            if idx >= len(lst):
                idx = len(lst) - 1
            res.append(lst[idx])

            lst.remove(lst[idx])
        return res

    def get_validation_model_rel(self, tscbn, rel_remove):
        cross_edges = self.get_cross_edges(tscbn)
        n_cross_edges = len(cross_edges)
        remove = round(n_cross_edges * rel_remove)

        return self.get_validation_model_abs(tscbn, remove)

    def get_validation_model_abs(self, tscbn, abs_remove):
        cross_edges = self.get_cross_edges(tscbn)
        n_cross_edges = len(cross_edges)

        if abs_remove > n_cross_edges:
            print("#remove > #cross-edges !!!")
            abs_remove = n_cross_edges

        new_model = deepcopy(tscbn)

        # remove edges
        for i in range(abs_remove):
            edge = cross_edges.pop(np.random.randint(0, len(cross_edges)))

            # remove edge from E
            new_model.E.remove(edge)

            source, dest = edge

            # clean source
            #   children
            new_model.Vdata[source]["children"].remove(dest)

            # clean dest
            #   parents
            parent_index = new_model.Vdata[dest]["parents"].index(source)
            new_model.Vdata[dest]["parents"].remove(source)
            if not new_model.Vdata[dest]["parents"]:
                new_model.Vdata[dest]["parents"] = None
                # cprobs
                cummulative_sum = np.array([0.0] * len(new_model.Vdata[dest]["vals"]))
                for removed_value in new_model.Vdata[source]["vals"]:
                    prev_given = "['" + removed_value + "']"
                    cummulative_sum += new_model.Vdata[dest]["cprob"][prev_given]
                new_model.Vdata[dest]["cprob"] = np.divide(cummulative_sum, cummulative_sum.sum())
            else:
                # cprobs
                new_cprob = {}
                value_lists = []
                # compute a list of all value lists of the parents (without the removed parent)
                for parent in new_model.Vdata[dest]["parents"]:
                    value_lists.append(new_model.Vdata[parent]["vals"])
                # ... to generate a list of all value combinations of (new) parents
                all_given_combinations = list(itertools.product(*value_lists))

                # for every combination, average over cprob
                for _given in all_given_combinations:
                    given = list(_given)
                    new_given = "[" + ", ".join(["'" + value + "'" for value in given]) + "]"
                    cummulative_sum = np.array([0.0] * len(new_model.Vdata[dest]["vals"]))
                    for removed_value in new_model.Vdata[source]["vals"]:
                        evidence = deepcopy(given)
                        evidence.insert(parent_index, removed_value)
                        prev_given = "[" + ", ".join(["'" + value + "'" for value in evidence]) + "]"
                        cummulative_sum += new_model.Vdata[dest]["cprob"][prev_given]
                    new_cprob[new_given] = np.divide(cummulative_sum, cummulative_sum.sum())

                new_model.Vdata[dest]["cprob"] = new_cprob

            #   type_parents
            new_model.Vdata[dest]["type_parents"]["d"].remove(source)

        return new_model

    @staticmethod
    def get_cross_edges(tscbn):
        cross_edges = [e for e in tscbn.E if e[0].rsplit("_", 1)[0] != e[1].rsplit("_", 1)[0] and not (
                str.startswith(e[0], "dL_") or str.startswith(e[1], "dL_"))]

        return cross_edges

    @staticmethod
    def set_CPDs(tscbn):
        for vertex in [v for v in tscbn.V if not str.startswith(v, "dL_")]:
            vdata = tscbn.Vdata[vertex]

            n_vals = len(vdata["vals"])

            dominant_value = np.random.randint(0, n_vals)
            if vdata["parents"]:
                for given in vdata["cprob"]:
                    vdata["cprob"][given] = ModelGenerator._random_cpd(n_vals, dominant_value)
            else:
                vdata["cprob"] = ModelGenerator._random_cpd(n_vals, dominant_value)

    @staticmethod
    def _random_cpd(length, index_of_max):
        # probabilty at index_of_max is in [0.6, 1.0]
        max_probability = 0.6 + np.random.rand() * 0.4

        probs = np.array([np.random.rand() for _ in range(length - 1)])

        probs = np.divide(probs, probs.sum()) * (1.0 - max_probability)
        result = np.insert(probs, index_of_max, max_probability)

        return result