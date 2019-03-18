import numpy as np
import pyModelChecking.CTL as CTL

from collections import defaultdict

from spec_mining.utility.levenshtein import Levenshtein
from spec_mining.specification.regex_specification import RegexSpec, SymbolType

from spec_mining.mining.miner import Miner


class CBMiner(Miner):

    def __init__(self, bn1, bn2):
        self.bn1 = bn1
        self.bn2 = bn2

        if not bn1:
            if not bn2:
                raise ValueError("At least one TSCBN is necessary for instantiation")
            bn1 = bn2
        elif not bn2:
            bn2 = bn1

        self.model_1 = bn1.to_Kripke()
        self.paths_1 = bn1.get_filtered_paths()

        self.model_2 = bn2.to_Kripke()
        self.paths_2 = bn2.get_filtered_paths()

    def start(self, evaluation=False):
        specs = self.mine_bn_specs(evaluation)

        if evaluation:
            return specs

        # if specs:
        #     if not os.path.exists("output/"):
        #         os.makedirs("output/")
        #     with open("output/mined_specs_{}.txt".format(self.get_time_string()), "w") as out:
        #         for spec in specs:
        #             out.write("{:.4f}, {}\n".format(spec[0], spec[1]))

        return [spec[1] for spec in specs]

    def mine_bn_specs(self, evaluation):
        """
        Starts the Specification Mining Process by Checking the Satisfiability of paths represented
        as regular expressions in a second TSCBN. If a path does not fit into the second TSCBN,
        similar paths are merged iteratively to the original path and re-checked.

        :param evaluation:
        :param N:
        :return:
        """
        fps = 0  # evaluation - counter for false positives
        specs = []
        specs_merge_sets = []
        confusion_matrix = Levenshtein().levenshtein_confusion_matrix(self.paths_1)

        # perform for every path (or at max N) specification mining iterations
        for i in range(len(confusion_matrix)):

            spec = RegexSpec(self.paths_1[i])

            merge_list = []
            distances = set(confusion_matrix[i])
            for index in [idx for x in [np.where(confusion_matrix[i] == d)[0] for d in distances] for idx in x]:
                merge_list.append(index)
                merge_set = set(merge_list)
                if any([_set.issubset(merge_set) for _set in specs_merge_sets]):
                    # Rule is redundant
                    break

                Xable = self._get_Xable_flags(index)
                spec.merge(self.paths_1[index], Xable)

                # terminate spec if it gets too loose - check via spec metrics
                metrics = self._get_spec_metrics(spec)
                if not self._check_spec_metrics(metrics):
                    break

                # translate to CTL formula
                phi = spec.to_CTL()

                if phi is None:
                    # Error during CTL translation
                    break

                # perform network checking
                sat_states_model2 = CTL.modelcheck(self.model_2, phi)

                if sat_states_model2:
                    try:
                        eval_bn_metrics = self._metric_in_eval_model(spec, self.bn2, sat_states_model2)

                        if not eval_bn_metrics:
                            # print("empty eval")
                            break

                        if np.min(eval_bn_metrics) < (1 - self.bn1.probability_threshold):
                            try:
                                LTL_spec = spec.to_LTL()
                            except:
                                print("LTL translation error")
                                break

                            if not LTL_spec is None:
                                supersets = [index for index, _set in enumerate(specs_merge_sets) if merge_set.issubset(_set)]
                                if supersets:
                                    specs = [s for i, s in enumerate(specs) if i not in supersets]
                                    specs_merge_sets = [s for i, s in enumerate(specs_merge_sets) if i not in supersets]

                                if evaluation:
                                    if metrics["n_matching_strings"] - len(merge_set) > 0:
                                        fps += 1

                                spec_probability = 1 - np.average([p["metric"] for p in [self.bn1.paths[i] for i in merge_list]])

                                specs.append((spec_probability, LTL_spec))
                                specs_merge_sets.append(merge_set)

                    except SpecException as e:
                        # print("SpecException occured")
                        break
                    break

        if evaluation:
            specs = [spec for probability, spec in specs]
            return [fps, specs]
        else:
            specs.sort(key=lambda x: x[0], reverse=True)

            return specs

    @staticmethod
    def _check_spec_metrics(metrics):
        """
        checks the metrics of a spec and returns either True if the spec is considered good enough
        or False if the spec should not be processed any more

        :param metrics:
        :return:
        """
        if metrics["rel_matches"] < 0.50:
            # L().log.info("Spec Metric violated: rel_matches = {:0.2f}%".format(metrics["rel_matches"]))
            return False

        # as we want around 40-50% exact matches and not too big Option/Group symbols (around 2.5 average max)
        # metric "n_matching_strings" should not exceed 2.5**(length / 2)
        threshold = 2.5**(metrics["length"] >> 1)
        if metrics["n_matching_strings"] > threshold:
            return False

        return True

    def _metric_in_eval_model(self, spec, eval_model, sat_states):
        """
        Computes the metric a specification has in the second TSCBN.

        :param spec: [RegexSpec] specification to be checked
        :param sat_states: [list] list of states the specification is satisfied
        :return: [list] list with metrics
        """

        def _rec_metric_in_eval_model(i, prev_value, prev_vertex, i_of_prev, probabilities):
            """
            Sub-Method for metric computation

            Attention: Only valid for AdvSpec class!

            :param i:
            :param prev_value:
            :param prev_vertex:
            :param i_of_prev:
            :param probabilities:
            :return:
            """
            if i < 1:
                raise ValueError("Invalid argument value i: {}".format(i))

            if len(result_list) > eval_model.n_paths:
                # L().log.info("Spec matches > {} times - break computation".format(eval_model.n_paths * 0.4))
                raise SpecException("Stopping Path Metric Computation - too many hits")

            if i >= len(spec):
                # full Spec found in the TSCBN => save probability in result_list
                result_list.append(np.average(probabilities))
                return

            # put all possible successor values into a dictionary with
            # key = node name
            # values = node's values
            next_symbols = defaultdict(lambda: [])
            key_list = [*spec[i].symbol]
            if spec[i].type == SymbolType.Letter:
                next_signal, next_value = key_list[0].rsplit(":", 1)
                next_symbols[next_signal].append(next_value)
            else:
                for symbol in key_list:
                    next_signal, next_value = symbol.rsplit(":", 1)
                    next_symbols[next_signal].append(next_value)

            result_list_before = len(result_list)

            # iterate over all children of the current node
            # if the child name starts with one of the keys in the successor dictionary (next_symbols)
            possible_next_children = [c for c in eval_model.Vdata[prev_vertex]["children"] if
                                      any([key for key in next_symbols if c.startswith(key)])]
            for child in possible_next_children:
                # edit key (depending on consistent naming)
                if child in next_symbols:
                    key = child
                else:
                    key = child.rsplit("_", 1)[0]

                # iterate over all values of the successor that are listed in the next_symbols dictionary
                for value in [v for v in eval_model.Vdata[child]["vals"] if v in next_symbols[key]]:
                    # compute edge weight between vertex and child (with values of spec)
                    probabilities.append(self._get_E_weight(eval_model, prev_vertex, prev_value, child, value))

                    # recursively call function for every child
                    _rec_metric_in_eval_model(i + 1, value, child, i, probabilities)

                    # pop last probability (of child) to make room for next child
                    probabilities.pop()

            # for Options, additionally call function recursively with next symbol - ommitting the current symbol
            # = taking the ? into account
            if spec[i].type == SymbolType.Option:
                _rec_metric_in_eval_model(i + 1, prev_value, prev_vertex, i_of_prev, probabilities)

            result_list_after = len(result_list)

            # Until allows options to occur multiple times, e.g.
            # a U b is satisfied by: b, ab, aab, aaab, ... (and so on)
            # Thus, go to next node of 'myself' if result list did not grow
            if spec[i_of_prev].type == SymbolType.Option and result_list_before == result_list_after:
                prev_signal = prev_vertex.rsplit("_", 1)[0]
                prev_index = int(prev_vertex.rsplit("_", 1)[1])
                next_myself = "{}_{}".format(prev_signal, prev_index + 1)
                if next_myself in eval_model.Vdata[prev_vertex]["children"]:
                    probabilities.append(
                        self._get_E_weight(eval_model, prev_vertex, prev_value, next_myself, prev_value))
                    _rec_metric_in_eval_model(i, prev_value, next_myself, i_of_prev, probabilities)
                    probabilities.pop()
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End of Subdefinition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #

        result_list = []

        non_options = [i for i, s in enumerate(spec) if s.type != SymbolType.Option]

        if not non_options:
            return [1]

        first_index = min(non_options)
        last_index = max(non_options)

        if first_index >= last_index:
            return [1]

        spec = spec[first_index: last_index + 1]

        # for every satisfying state, check where paths exist that match the regex
        for istate in sat_states:

            first_vertex, first_value = istate.rsplit(":", 1)
            value_name = eval_model.Vdata[first_vertex]["vals"][int(first_value)]

            _rec_metric_in_eval_model(1, value_name, first_vertex, 0, [])

        return result_list


    @staticmethod
    def _get_E_weight(model, from_vertex, from_value, to_vertex, to_value):
        """
        Fetches the edge weight between two vertices with given values from the E_weights dictionary
        of the second TSCBN. If the desired edge does not exist in the dictionary, the absolute
        probability of the destination vertex holding the given value is returned.

        :param from_vertex: [string] name of the source vertex
        :param from_value: [string] value of the source vertex
        :param to_vertex: [string] name of the destination vertex
        :param to_value: [string] value of the destination vertex
        :return: [numeric] edge weight between given vertices with given values or absolute probability of
        destination vertex holding the given value if edge does not exist.
        """
        from_node = "{0}:{1}".format(from_vertex, model.Vdata[from_vertex]["vals"].index(from_value))
        to_node = "{0}:{1}".format(to_vertex, model.Vdata[to_vertex]["vals"].index(to_value))

        # look in E_weights for edge weight
        if from_node in model.E_weights:
            if to_node in model.E_weights[from_node]:
                return model.E_weights[from_node][to_node]

        # if edge is not in E_weights (e.g. no from_vertex known) return absolute probability
        to_value_index = model.Vdata[to_vertex]["vals"].index(to_value)
        return model.absolute_probs[to_vertex][to_value_index]

    def _get_Xable_flags(self, index):
        # returns a list of Xable flags for a given path (passed by the index of the path)
        flags = [False]
        path = self.bn1.get_paths()[index]
        for from_node, to_node in zip(path[:-1], path[1:]):
            from_v = from_node.rsplit(":", 1)[0]
            to_v = to_node.rsplit(":", 1)[0]
            if from_v in self.bn1.E_histogram:
                if to_v in self.bn1.E_histogram[from_v]:
                    flags.append(self.bn1.E_histogram[from_v][to_v])
                    continue
            flags.append(False)

        return flags


class SpecException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
