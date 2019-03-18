from spec_mining.mining_graph.mining_graph import MiningGraph

from collections import defaultdict
from operator import itemgetter

import math


class MaxAverageMiningGraph(MiningGraph):

    def __init__(self, tscbn):
        super().__init__(tscbn)

    # Yen Algorithm for finding the K shortest paths from "start:0" to "terminal:0"
    def path_computation(self, min_prob_threshold=0.8, min_path_length=2, min_signals_per_path=1, **kwargs):
        self.probability_threshold = min_prob_threshold
        max_metric_threshold = MiningGraph._p2metric(min_prob_threshold)
        shortest_path = self._dijkstra_min_average("start:0")

        A = [{"path"  : shortest_path["path"],
              "metric": shortest_path["metric"]}]
        B = []

        if not A[0]["path"]: return A

        if A[0]["metric"] > max_metric_threshold:
            return []

        while True:
            for i in range(0, len(A[-1]["path"]) - 1):
                spur_node = A[-1]["path"][i]
                root_path = A[-1]["path"][:i+1]

                removed_spur_node_targets = []
                removed_edges = []
                for shorter_path in A:
                    current_path = shorter_path["path"]
                    if len(current_path) > i and root_path == current_path[:i + 1]:
                        edge = "{0},{1}".format(current_path[i], current_path[i + 1])
                        if edge not in removed_edges:
                            removed_edges.append(edge)
                            removed_spur_node_targets.append(current_path[i + 1])

                root_path_edges = max(0, len(root_path) - 2)
                root_path_cost = self._get_root_path_distance(root_path)
                root_path_avg = 0 if root_path_edges == 0 else root_path_cost / root_path_edges

                spur_path = self._dijkstra_min_average(spur_node, root_path_edges, root_path_avg, removed_edges, removed_spur_node_targets)

                if spur_path["path"]:
                    total_path = root_path[:-1] + spur_path["path"]

                    candidate_path = {"path": total_path, "metric": spur_path["metric"]}

                    if not candidate_path in B:
                        B.append(candidate_path)

            if len(B):
                B = sorted(B, key=itemgetter("metric"))
                b = B.pop(0)
                if b["metric"] > max_metric_threshold:
                    break
                A.append(b)
            else:
                # empty B queue -> terminate
                break

        shortest_paths = [a for a in A if self._is_accepting_path(a, min_path_length, min_signals_per_path, max_metric_threshold)]

        self.paths = shortest_paths
        return self.paths

    # Dijkstra algorithm
    def _dijkstra_min_average(self, start_node, root_path_edges=0, root_path_avg=0, removed_edges=[], removed_start_node_targets=[]):
        reachable_vertices = self._reachable_vertices(start_node.rsplit(":", 1)[0], removed_start_node_targets)
        topological_order = self._topological_sort_vertices(reachable_vertices)

        _dict_distance = defaultdict(lambda: defaultdict(lambda: math.inf))
        _dict_previous = defaultdict(lambda: defaultdict(lambda: None))
        _dict_distance[start_node][root_path_edges] = root_path_avg

        final_edges = 0
        final_distance = math.inf

        while topological_order:
            source_vertex = topological_order.pop(0)
            # for all values of the source node
            for i in range(len(self.Vdata[source_vertex]["vals"])):
                source_node = source_vertex + ":" + str(i)
                # edges = n_edges[source_node]

                for dest_node, weight in self.E_weights[source_node].items():
                    if "{0},{1}".format(source_node, dest_node) in removed_edges:
                        weight = math.inf

                    for _edges, _metric in _dict_distance[source_node].items():
                        if dest_node == "terminal:0":
                            new_distance = (_metric * _edges + weight) / _edges
                        else:
                            new_distance = (_metric * _edges + weight) / (_edges + 1)

                        if source_node == "start:0" or dest_node == "terminal:0":
                            _edges -= 1

                        if _dict_distance[dest_node][_edges + 1] > new_distance:
                            _dict_distance[dest_node][_edges + 1] = new_distance
                            _dict_previous[dest_node][_edges + 1] = source_node

                        if dest_node == "terminal:0" and final_distance > new_distance:
                            final_edges = _edges + 1
                            final_distance = new_distance

        return {"path": self._path_from_previous(_dict_previous, final_edges, start_node),
                "metric": _dict_distance["terminal:0"][final_edges]}

    @staticmethod
    def _path_from_previous(shortest_path_tree, n_edges, start, end="terminal:0"):
        """
        Computes a path from a start node to an end node given the shortest path tree

        :param shortest_path_tree: [list] list of previous nodes in the shortest path tree
        :param start: [string] name of the start node of the path
        :param end: [string] name of the end node of the path, default: terminal:0
        :return: [list] shortest path from start to end
        """
        path = [end]
        current_node = end
        while current_node != start:
            current_node = shortest_path_tree[current_node][n_edges]
            if n_edges > 0 and len(path) > 1:
                n_edges -= 1
            if current_node is None:
                return []
            path.insert(0, current_node)

        return path

    def _get_root_path_distance(self, root_path):
        total_distance = 0
        for from_node, to_node in zip(root_path[:-1], root_path[1:]):
            total_distance += self.E_weights[from_node][to_node]
        return total_distance

    def _reachable_vertices(self, start_vertex, removed_start_node_targets):
        vertices = [start_vertex]

        _unreachable = defaultdict(lambda: 0)
        for removed_target in removed_start_node_targets:
            key = removed_target.rsplit(":",1)[0]
            _unreachable[key] += 1

        unreachable = [child for child in self.Vdata[start_vertex]["children"]
                       if _unreachable[child] == len(self.Vdata[child]["vals"])]

        queue = [child for child in self.Vdata[start_vertex]["children"] if child not in unreachable]
        while queue:
            vertex = queue.pop(0)
            if vertex not in vertices:
                vertices.append(vertex)
            queue.extend(self.Vdata[vertex]["children"])
        return vertices

    def _is_accepting_path(self, candidate_path, min_length, min_signals, max_metric):
        """
        Checks if a path fulfills certain conditions passed as arguments.
        Conditions are:
            minimum path length: minimum number of nodes the path is made of
            minimum number of signals: minimum number of different signals a path must contain
                                        (only checked if consistent_naming = True)
            maximum metric: maximum metric (length/ weight) a path must not exceed

        :param candidate_path: [list] path to be checked
        :param min_length: [int] minimum number of nodes
        :param min_signals: [int] minimum number of different signals a path must contain
        :param max_metric: [float] maximum metric a path must not exceed
        :return: [boolean] True if all conditions are fullfilled, False otherwise
        """
        p = candidate_path["path"][1:-1]
        if len(p) < min_length:
            return False

        # number of signals can only be checked if vertices obey consistent naming
        # (see _setup_mining_tscbn method)
        if self.consistent_naming:
            if len(set([node.rsplit("_", 1)[0] for node in p])) < min_signals:
                return False

        if candidate_path["metric"] > max_metric:
            return False

        return True
