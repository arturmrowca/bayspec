from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import add

from _include.m_libpgm.hybayesiannetwork import HyBayesianNetwork
from _include.m_libpgm.nodedata import NodeData
from network.tscbn import TSCBN
from pyModelChecking import *

import copy
import math
import numpy as np
import random
import re


class MiningGraph(HyBayesianNetwork, metaclass=ABCMeta):

    def __init__(self, source_tscbn : TSCBN):
        super().__init__()

        self._setup_mining_tscbn(source_tscbn)

        nd = NodeData()
        nd.Vdata = self.Vdata
        nd.entriestoinstances()
        self.nodes = {k: nd.nodes[k] for k in self._topological_sort_vertices(nd.nodes)}

        assert sorted(self.V) == sorted(self.Vdata.keys()), "Vertices do not match Vdata keys."

        self.absolute_probs = self._calc_absolute_probs()

        self.E_weights = self._calc_edge_weights()

        self.E_histogram = self._init_Xable_flags()

        self.paths = []

        self.probability_threshold = 0

        self.extended_paths = []

    @abstractmethod
    def path_computation(self, **kwargs) -> list:
        """
        Computes the shortest paths by some metric of the TSCBN

        :param kwargs: optional named arguments for implementations in subclasses
        :return: [list] list of dictionary entries for a path with the mandatory keys:
                "path": list of nodes representing the path
                "metric": metric of the path
        """

    def to_Kripke(self) -> Kripke:
        """
        Creates a Kripke structure of the TSCBN used for network checking

        :return: [Kripke] Kripke structure representing the TSCBN
        """
        S0 = ["start:0"]
        States = ["{0}:{1}".format(v, i) for v in self.V for i in range(len(self.Vdata[v]["vals"]))]

        Relations = [("{0}:{1}".format(e[0], i), ("{0}:{1}".format(e[1], j)))
                     for e in self.E
                     for i in range(len(self.Vdata[e[0]]["vals"]))
                     for j in range(len(self.Vdata[e[1]]["vals"]))]

        Relations.append(("terminal:0", "terminal:0"))

        if self.consistent_naming:
            Labels = {"{0}:{1}".format(v, i): {"{0}:{1}".format(v.rsplit("_", 1)[0], value)}
                      for v in self.V
                      for i, value in enumerate(self.Vdata[v]["vals"])}
        else:
            Labels = {"{0}:{1}".format(v, i): {"{0}:{1}".format(v, value)}
                      for v in self.V
                      for i, value in enumerate(self.Vdata[v]["vals"])}

        return Kripke(S=States, S0=S0, R=Relations, L=Labels)

    def _setup_mining_tscbn(self, source_tscbn: TSCBN):
        """
        Transforms the TSCBN to be used for shortest path search.
        Sets up the class members.
        First method in the constructor.

        Method consists of several procedures.
        1) Reduce
            The "dL_"-vertices and edges to "dL_"-vertices are removed from self.V and self.E
            The "dL_"-vertex entries from self.Vdata are removed and stored in self.dL_Vdata
            The "dL_"-vertices are removed from children lists within self.Vdata
        2) Vertex Naming Check
            The self.consistent_naming flag is set due to _check_vertex_naming (see for more information)
        3) NaN Treatment
            NaNs in the probabilities are replaced by numeric values due to _nan_to_numeric method
        4) Extend
            Depending on self.consistent_naming the root and leaf vertices are computed.
            The artificial vertices "start" and "terminal" are introduced. Both have one single value
            defined by the variable "single_artificial_value".
            Edges from the start vertex to root vertices are added.
            Edges from leaf vertices to terminal vertex are added.
            The parents, children, cprob and vals keys of start, terminal, root and leaf vertices are
            added/modified.

        :param source_tscbn: [TSCBN] input TSCBN to be transformed
        :return: [void]
        """
        # --------------- REDUCE ---------------
        vertices = []
        # sort out edges from or to time vertices
        edges = [e for e in source_tscbn.E if not str.startswith(e[0], "dL_") and not str.startswith(e[1], "dL_")]
        for e in edges:
            if e[0] not in vertices:
                vertices.append(e[0])
            if e[1] not in vertices:
                vertices.append(e[1])

        self.V = copy.deepcopy(vertices)
        self.E = copy.deepcopy(edges)

        # clean the Vdata dictionary
        # first sort out time vertex keys
        vdata = {vertex: data for vertex, data in source_tscbn.Vdata.items() if vertex in vertices}
        dL_vdata = {vertex: data for vertex, data in source_tscbn.Vdata.items() if vertex not in vertices}
        # second clean children
        for vertex, data in vdata.items():
            data["children"] = [child for child in data["children"] if child in vertices]

        self.Vdata = copy.deepcopy(vdata)
        self.nodes = copy.deepcopy({node: data for node, data in source_tscbn.nodes.items() if node in vertices})

        self.dL_Vdata = copy.deepcopy(dL_vdata)

        # --------- VERTEX NAMING CHECK ---------
        # assert self._check_vertex_naming(), "Vertices do not have consistent names."
        self.consistent_naming = self._check_vertex_naming(self.V)

        # ------------ NaN TREATMENT ------------
        for vertex in self.V:
            if "cprob" in self.Vdata[vertex]:
                if self.Vdata[vertex]["parents"]:
                    for given, cprob in self.Vdata[vertex]["cprob"].items():
                        result = MiningGraph._nan_to_numeric(cprob)
                        self.Vdata[vertex]["cprob"][given] = result
                else:
                    result = MiningGraph._nan_to_numeric(self.Vdata[vertex]["cprob"])
                    self.Vdata[vertex]["cprob"] = result

        # --------------- EXTEND ---------------
        single_artificial_value = "0"
        root_vertices = []
        leaf_vertices = []

        if self.consistent_naming:
            # when consistent_naming is True:

            # collect the number of vertices per signal
            signals = defaultdict(int)
            for v in self.V:
                signals[v.rsplit("_", 1)[0]] += 1

            # root vertices are all signal vertices with index 0
            # leaf vertices are the last vertices of signals, if a signal has at least 2 vertices
            # otherwise that signal has no vertex with an edge to the terminal vertex
            for signal, num_vertices in signals.items():
                root_vertices.append("{0}_0".format(signal))
                if num_vertices > 1:
                    leaf_vertices.append("{0}_{1}".format(signal, num_vertices - 1))

        else:
            # when there is no consistent naming
            # root vertices are vertices without parents
            # leaf vertices are vertices witout children
            root_vertices = self._get_root_vertices()
            leaf_vertices = self._get_leaf_vertices()

        # artificial start vertex
        self.Vdata["start"] = dict()
        self.Vdata["start"]["vals"] = [single_artificial_value]
        self.Vdata["start"]["children"] = []
        self.Vdata["start"]["parents"] = None
        self.Vdata["start"]["cprob"] = np.array([1.0])
        self.Vdata["start"]["numoutcomes"] = 1
        self.Vdata["start"]["type"] = "discrete"

        for vertex in root_vertices:
            if self.Vdata[vertex]["parents"]:
                self.Vdata[vertex]["parents"].append("start")
                cprob_dict = dict()
                for given, probs in self.Vdata[vertex]["cprob"].items():
                    cprob_dict["{}, '0']".format(given[:-1])] = self.Vdata[vertex]["cprob"][given]
                self.Vdata[vertex]["cprob"] = cprob_dict
            else:
                self.Vdata[vertex]["parents"] = ["start"]
                probability_distribution = self.Vdata[vertex]["cprob"]
                self.Vdata[vertex]["cprob"] = dict()
                self.Vdata[vertex]["cprob"]["['" + single_artificial_value + "']"] = probability_distribution
            self.Vdata["start"]["children"].append(vertex)
            self.E.append(["start", vertex])

        self.V.insert(0, "start")

        # artificial terminal vertex
        self.Vdata["terminal"] = dict()
        self.Vdata["terminal"]["vals"] = [single_artificial_value]
        self.Vdata["terminal"]["children"] = []
        self.Vdata["terminal"]["parents"] = []

        # Normally, cprob key gets a dictionary with every combination of values of its parents
        # but this quickly explodes and results in a bottleneck.
        # Therefore, a defaultdict is used and returns [1.0]
        self.Vdata["terminal"]["cprob"] = defaultdict(lambda: np.array([1.0]))
        self.Vdata["terminal"]["numoutcomes"] = 1
        self.Vdata["terminal"]["type"] = "discrete"

        for leaf in leaf_vertices:
            if "children" in self.Vdata[leaf]:
                self.Vdata[leaf]["children"].append("terminal")
            else:
                self.Vdata[leaf]["children"] = ["terminal"]
            self.Vdata["terminal"]["parents"].append(leaf)
            self.E.append([leaf, "terminal"])

        self.V.append("terminal")
        self.V = self._topological_sort_V()
        # --------------- NUMBER OF PATHS ---------------
        self.n_paths = self._number_of_dag_paths()

    @staticmethod
    def _check_vertex_naming(V) -> bool:
        """
        Checks if any vertex contains forbidden characters. Otherwise error is raised.
        Checks if all vertices of param "V" obey consistent naming scheme
        scheme: <signal-name>_<vertex-number>
        vertex numbers have to range from 0 to number of vertices of that signal

        :param V: [list] list of vertices to be checked
        :return: [boolean] True if no vertex name contains a forbidden character and all vertices follow
        the specified naming scheme
        """
        # check for forbidden characters
        assert not any([re.search(",", v) for v in V]), "Vertex names contain forbidden character"

        # check for naming scheme: <signal-name>_<vertex-number>
        if not all([re.search(".+_[0-9]+", v) for v in V]):
            return False

        # check if vertex-numbers per signal start from 0 and do not leave out any number
        signals = defaultdict(list)
        for v in V:
            signal, index = v.rsplit("_", 1)
            signals[signal].append(int(index))

        for k, v in signals.items():
            for i, x in enumerate(sorted(v)):
                # check if i-th vertex has number i
                if i != x:
                    return False
        return True

    @staticmethod
    def _nan_to_numeric(vector, strategy="tozero1"):
        """
        Replaces NaNs in a vector to numeric values depending on the selected strategy
        none    : don't do anything
        split   : split the remaining (1 - p_sum) uniformly to the nan's
        tozero1 : set nan to 0, split the rest (1 - p_sum) uniformly to the numeric probabilites
        tozero2 : set nan to 0, split the rest (1 - p_sum) weighted by their ratio to the numeric probabilites

        for all strategies: if vector contains only NaNs -> uniform distribution

        :param vector: [list] probability distribution list to be treated
        :param strategy: [string] chosen strategy to treat NaNs
        :return: [list] corrected vector
        """
        num_nans = np.count_nonzero(np.isnan(vector))
        if num_nans > 0:
            p_sum = sum([x for x in vector if not math.isnan(x)])
            if p_sum > 0:
                # there is at least one numeric value in the distance list
                if strategy == "split":
                    vector = [(1.0 - p_sum) / num_nans if math.isnan(x) else x for x in vector]
                elif strategy == "tozero1":
                    vector = [x + (1.0 - p_sum) / (len(vector) - num_nans) if not math.isnan(x) else 0.0 for x in
                              vector]
                elif strategy == "tozero2":
                    vector = [x + (1.0 - p_sum) * x / p_sum if not math.isnan(x) else 0.0 for x in vector]
            else:
                # only nan's in the list -> uniform distribution
                vector = [1.0 / len(vector)] * len(vector)

        return vector

    @staticmethod
    def _p2metric(p):
        """
        Calculates the edge weight for the resulting MiningGraph given a probability

        :param p: probability to be transformed into an edge weight
        :return: edge weight
        """
        if p <= 0:
            return 1

        if p >= 1:
            return 0

        return 1 - p

    @staticmethod
    def _metric2p(m):
        return MiningGraph._p2metric(m)

    def _get_root_vertices(self):
        """
        Returns all vertices of a TSCBN without parents (root vertices)

        :return: [list] list of vertices without parents, [] otherwise
        """
        return [v for v in self.V if not self.Vdata[v]["parents"]]

    def _get_leaf_vertices(self):
        """
        Returns all vertices of a TSCBN without children (leaf vertices)

        :return: [list] list of vertices without children, [] otherwise
        """
        return [v for v in self.V if not self.Vdata[v]["children"]]

    def _topological_sort_V(self):
        """
        Returns a list of the vertices in the TSCBN in topological order

        :return: [list] topologically ordered list of the TSCBN's vertices
        """
        return self._topological_sort_vertices(self.V)

    def _topological_sort_vertices(self, vertices):
        """
        Returns a list of topological ordered vertices passed as argument

        :param vertices: [list] list of vertices that are ordered topologically
        :return: [list] topologically ordere list of passed vertices
        """
        stack = []
        visited = dict((v, False) for v in vertices)
        for vertex in vertices:
            if not visited[vertex]:
                self._rec_topological_sort(vertex, visited, stack)
        return stack

    def _rec_topological_sort(self, vertex, visited, stack):
        """
        Recursive function for computation of a topological order of vertices

        :param vertex: [string] name of the current vertex
        :param visited: [list] list of booleans indicating whether a vertex has already been visited
        :param stack: [list] resulting topological order
        :return: [void]
        """
        if visited[vertex]:
            return
        for child in [c for c in self.Vdata[vertex]["children"] if c in visited]:
            self._rec_topological_sort(child, visited, stack)
        visited[vertex] = True
        stack.insert(0, vertex)

    def _generic_travers(self, fct, *args) -> dict:
        """
        Traverses the TSCBN from root to leaf vertices and processes every vertex with a function
        passed as argument "fct". A vertex is only processed if all its parent vertices have been
        processed before. The result of each vertex is saved in a result dictionary with
        key = vertex-name, value = result of vertex processing function

        :param fct: [function] vertex processing function
        :param *args: additional optional arguments the vertex processing function might need
        :return: [dict] dictionary containing the result of the processing function per vertex
        """
        result_dict = defaultdict(int)
        queue = ["start"]
        while queue:
            vertex = queue.pop(0)
            # check if node has already been processed
            if vertex in result_dict:
                continue

            # check if all parents have been computed
            parents = [] if self.Vdata[vertex]["parents"] is None else self.Vdata[vertex]["parents"]
            check = [p for p in parents if p not in result_dict]
            if check:
                if not queue:
                    print("ERROR in generic_travers: endless loop (not all parents processed)")
                    raise SystemExit
                queue.append(vertex)
                continue

            # compute desired result
            result = fct(vertex, result_dict, args)
            # write result into result dictionary
            result_dict[vertex] = result
            # add children to queue
            queue.extend(self.Vdata[vertex]["children"])

        # delete entries where value is None (because caller function might not work otherwise)
        return {k: v for k, v in result_dict.items() if v is not None}

    def _calc_absolute_probs(self) -> dict:
        """
        Calculates the absolute probabilities of a vertex' values, for all vertices of the TSCBN
        by calling the _generic_travers method with the function defined below

        :return: [dict] dictionary containing a list of probabilities for every vertex in the TSCBN
        e.g. TSCBN with 2 vertices ("V1" & "V2"), each having 2 possible values
        {"V1" : [0.26, 0.74], "V2": [0.19, 0.81]}
        """

        def absolute_probs_fct(vertex, result_dict, *args):
            """
            Function to be passed as parameter to the _generic_travers method to calculate
            the absolute probabilites of each value per vertex.

            :param vertex: [string] vertex-name of the vertex to be processed
            :param result_dict: [dict] dictionary to look for previous results
            :param args: #notused
            :return: [list] list of absolute probabilities of a vertex' values for the passed vertex

            Notes:
            The absolute probability of a value v of vertex V is calculated by
            P(V=v) = SUM[x_1] SUM[x_2] ... SUM[x_n] P(V=v|p_1=x_1,p_2=x_2,...,p_n=x_n)*P(p_1=x_1)*P(p_2=x_2)*...*P(p_n=x_n)
            where
                p_i is the i-th parent of vertex V
                P(p_i=x_i) is the absolute probability that parent p_i equals x_i
                P(V=v|p_1=x_1,p_2=x_2,...,p_n=x_n) is the probability of v given specific parent values

            example:
                vertex V has parents p1 and p2, all are binary
                P(V=v) = P(V=v|p1=0,p2=0)*P(p1=0)*P(p2=0) + P(V=v|p1=0,p2=1)*P(p1=0)*P(p2=1) +
                         P(V=v|p1=1,p2=0)*P(p1=1)*P(p2=0) + P(V=v|p1=1,p2=1)*P(p1=1)*P(p2=1)
            """
            if vertex == "start" or vertex == "terminal":
                return np.array([1.0])
            else:
                total_prob = [0.0] * len(self.Vdata[vertex]["vals"])
                # loop over all probability distributions for specific parent values
                for key, val in self.Vdata[vertex]["cprob"].items():
                    # examples: key = "['o0_0', 'o1_1', 'o2_1']
                    # parse the key to get all given values that make up the 'cprob' key
                    # key[1:-1] removes the opening and closing bracket
                    # split(",") splits the values
                    # strip()[1:-1] removes leading and trailing whitespace and single quotes
                    key_list = [k.strip()[1:-1] for k in key[1:-1].split(",")]
                    i = 0
                    prob = 1.0
                    for parent in self.Vdata[vertex]["parents"]:
                        # fetch absolute probability of parent having the specific value
                        parent_value = key_list[i]
                        parent_value_index = self.Vdata[parent]["vals"].index(parent_value)
                        # accumulative multiplication of single value probabilities (last n-1 factors in formula above)
                        prob *= result_dict[parent][parent_value_index]
                        i += 1
                    # finally, multiply product of single value probabilities with probability distribution
                    # of vertex given specific parent values
                    total_prob = list(map(add, total_prob, [p * prob for p in val]))
                return np.array(total_prob)
        # ------------------------- END OF LOCAL FUNCTION DEFINITION -------------------------
        return self._generic_travers(absolute_probs_fct)

    def _calc_edge_weights(self):
        """
        Calculates the edge weight between all nodes in the TSCBN.
        Returns a dictionary where the (source) node names are the keys, the values are dictionaries
        with the (destination) node names as keys, the values are the edge weights.

        Node names consist of vertex names and their values (value indices).

        One edge in self.E results in multiple edges in the calculated dictionary.
        For every value of the source vertex an edge points to every value of the destination vertex.
        E.g. for the edge ["vertex_a", "vertex_b"] in self.E (where both vertices have 2 values) the
        resulting dictionary contains (amongst other)
        {"vertex_a:0": {"vertex_b:0": 0.12, "vertex_b:1": 1.03},
         "vertex_a:1": {"vertex_b:0": 1.60, "vertex_b:1": 0.45},
        }

        Example result:
        { "A0:0": {"A1:0": 0.37, "A1:1": 1.08, "B1:0": 0.63, "B1:1": 1.92},
          "A0:1": {"A1:0": 1.45, "A1:1": 0.35, "B1:0": 1.42, "B1:1": 0.77},
          "B0:0": {"B1:0": 0.78, "B1:1": 0.73},
          "B0:1": {"B1:0": 1.18, "B1:1": 0.25},
          ...
        }

        Lookup for an edge weight for the edge from node "from_node" to node "to_node" (in the resulting
        dictionary edge_weights) is:
        edge_weight = edge_weights[from_node][to_node]

        :return: [dict] dictionary with edge weights
        """

        return {"{0}:{1}".format(v,i) :
                    {"{0}:{1}".format(child,j): self._p2metric(weight)  # log_of_P(weight)
                     for child in self.Vdata[v]["children"]
                     for j, weight in enumerate(self._distance("{0}:{1}".format(v, i), child))}
                for v in self.V
                for i in range(len(self.Vdata[v]["vals"]))}

    def _distance(self, source_node, dest_vertex):
        """
        Calculates the edge distances from a source node to all nodes of a destination vertex.

        :param source_node: [string] source node of the edges whose distances are calculated
        :param dest_vertex: [string] destination vertex (representing all destination nodes) of the edges
            whose distances are calculated
        :return: [list] list of edge distances from a source node to the nodes of a destination vertex

        Notes:
            Case 1: source_vertex is the only parent of dest_vertex
                No marginalization needed.
                Edge distances from source_node to dest_nodes is the list of probabilities of dest_nodes
                given source_node, which is an easy lookup: distances = self.Vdata[dest_vertex]["cprob"][evidence]
                where evidence is the value of source_node
            Case 2: dest_vertex has > 1 parents
                Marginalization needed, because only 1 parent value is known (source_node).
                Edge distances are computed similar to _calc_absolute_probs but with the difference
                that here one value is given (source_node)
                distances = SUM[x_1]... SUM[x_k-1] SUM[x_k+1] ... SUM[x_n] P(V|p_1=x_1,...,p_k=y,...,p_n=x_n)*P(p_1=x_1)...*P(p_k-1=x_k-1)*P(p_k+1=x_k+1)*...*P(p_n=x_n)
                where
                    p_i is the i-th parent of vertex V
                    k is the index of source_vertex within the parents of V
                    y is the value of parent p_k
                    P(V|evidence) is the (discrete) conditional probability distribution of the values of V given evidence
                    P(p_i=x_i) is the absolute probability that parent p_i equals x_i

        Examples:
            1)  compute distances between source_node X:0 and dest_vertex Y, X is only parent of Y
                distances = P(Y|X=0)
                # distances: [0.34, 0.29]
                # i.e. distance( X:0 --> Y:0 ) = 0.34
                #      distance( X:0 --> Y:1 ) = 0.29

            2)  compute distances between source_node X:0 and dest_vertex Y, Y has parents X and U (binary)
                distances = P(Y|X=0,U=0)*P(U=0) + P(Y|X=0,U=1)*P(U=1)
                # distances: [0.34, 0.29]
                # i.e. distance( X:0 --> Y:0 ) = 0.34
                #      distance( X:0 --> Y:1 ) = 0.29
        """
        if source_node == "start:0":
            return [1.0] * len(self.Vdata[dest_vertex]["vals"])
        # shortcut for faster computation
        if dest_vertex == "terminal":
            return np.array([1.0])

        source_vertex, evidence_index = source_node.rsplit(":",1)
        evidence_index = int(evidence_index)

        dest_parents = self.Vdata[dest_vertex]["parents"]
        if len(dest_parents) == 1:
            # nice case: single parent
            # translate given index into given value
            evidence = self.Vdata[source_vertex]["vals"][evidence_index]
            evidence = "['" + evidence + "']"
            # look up probability of dest_vertex given the evidence value of source_vertex
            distances = self.Vdata[dest_vertex]["cprob"][evidence]
        else:
            # case multiple parents
            source_parent_index = dest_parents.index(source_vertex)
            source_value = self.Vdata[source_vertex]["vals"][evidence_index]
            distances = [0.0] * len(self.Vdata[dest_vertex]["vals"])

            # loop over all probability distributions for specific parent values
            # key = string of given parent values
            # val = probability distribution of dest_vertex given the values in key
            for key,val in self.Vdata[dest_vertex]["cprob"].items():
                key_list = [k.strip()[1:-1] for k in key[1:-1].split(",")]
                # only take cases where the value of the given vertex (source_vertex)
                # equals the value of that vertex in the key
                if key_list[source_parent_index] != source_value:
                    continue
                prob = 1.0
                i = 0
                for parent in dest_parents:
                    if i != source_parent_index:
                        parent_value = key_list[i]
                        parent_value_index = self.Vdata[parent]["vals"].index(parent_value)
                        prob *= self.absolute_probs[parent][parent_value_index]
                    i += 1

                distances = list(map(add, distances, [p * prob for p in val]))

        return distances

    def _init_Xable_flags(self):
        result = defaultdict(lambda: dict())
        for v in self.V:
            # for every node: with 70% choose one parent that receives a true Xable flag
            set_X = random.random() < 0.7
            if self.Vdata[v]["parents"] is None:
                continue
            for parent in self.Vdata[v]["parents"]:
                result[parent][v] = False
            if set_X:
                random_parent = random.choice(self.Vdata[v]["parents"])
                result[random_parent][v] = True
        return result

    def _number_of_dag_paths(self):
        paths = dict()
        result = self._n_paths("start", "terminal", paths)
        return result

    def _n_paths(self, v, target, paths):
        if v == target:
            return 1
        else:
            if v not in paths:
                paths[v] = np.sum((len(self.Vdata[v]["vals"]) * self._n_paths(child, target, paths))
                                  for child in self.Vdata[v]["children"])
            return paths[v]

    def get_paths(self):
        """
        Returns a list of paths(lists) filtered from the self.paths dictionary
        without first (start:0) and last (terminal:0) element

        example:
            self.paths = [{"path": ['start:0', 'signal_A_0:0', 'signal_B_1:1', 'terminal:0'], "metric": ...},
                          {"path": ['start:0', 'signal_X_0:2', 'signal_B_1:3', 'terminal:0'], "metric": ...}]

            returns: [ ['signal_A_0:0', 'signal_B_1:1'], ['signal_X_0:2', 'signal_B_1:3'] ]

        :return:
        """
        return [p["path"][1:-1] for p in self.paths]

    def get_filtered_paths(self):
        """
        Returns a list of paths(lists) filtered from the self.paths dictionary
        without first (start:0) and last (terminal:0) element,
        without the signal vertex numbers (if consistent_nameing = True)
        with value names instead of value indices

        example 1:
            self.paths = [{"path": ['start:0', 'signal_A_0:0', 'signal_B_1:1', 'terminal:0'], "metric": ...},
                          {"path": ['start:0', 'signal_X_0:2', 'signal_B_1:2', 'terminal:0'], "metric": ...}]
            self.Vdata = {"signal_A_0": {"vals": ['a0', 'a1', 'a2'], ...},
                          "signal_B_1": {"vals": ['b0', 'b1', 'b2'], ...},
                          "signal_X_0": {"vals": ['x0', 'x1', 'x2'], ...},
                          ...
                         }

            returns: [ ['signal_A:a0', 'signal_B:b1'], ['signal_X:x2', 'signal_B:b2'] ]

        example 2:
            self.paths = [{"path": ['start:0', 'signalX:0', 'signalY:1', 'terminal:0'], "metric": ...},
                          {"path": ['start:0', 'signalY:2', 'signalZ:2', 'terminal:0'], "metric": ...}]
            self.Vdata = {"signalX": {"vals": ['x0', 'x1', 'x2'], ...},
                          "signalY": {"vals": ['y0', 'y1', 'y2'], ...},
                          "signalZ": {"vals": ['z0', 'z1', 'z2'], ...},
                          ...
                         }

            returns: [ ['signalX:x0', 'signalY:y1'], ['signalY:y2', 'signalZ:z2'] ]

        :return: list of filtered paths
        """
        # return [["{0}:{1}".format(u.rsplit("_",1)[0], v) for u, v in s]
        #         for s in [[node.rsplit(":",1) for node in path] for path in [_d["path"][1:-1] for _d in self.paths]]]

        if self.consistent_naming:
            return [["{0}:{1}".format(u.rsplit("_", 1)[0], self.Vdata[u]["vals"][int(v)]) for u, v in s]
                    for s in [[node.rsplit(":", 1) for node in path] for path in self.get_paths()]]
        else:
            return [["{0}:{1}".format(u, self.Vdata[u]["vals"][int(v)]) for u, v in s]
                    for s in [[node.rsplit(":", 1) for node in path] for path in self.get_paths()]]
