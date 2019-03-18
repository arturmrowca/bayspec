import numpy as np

from spec_mining.utility.levenshtein import Levenshtein
from spec_mining.specification.regex_specification import RegexSpec
from spec_mining.mining.miner import Miner


class MBMiner(Miner):
    def __init__(self, bayesian_network):
        self.bn = bayesian_network
        self.paths = bayesian_network.get_filtered_paths()

    def start(self):
        specs = self.mine_bn_specs()

        # if specs:
        #     if not os.path.exists("output/"):
        #         os.makedirs("output/")
        #     with open("output/mined_specs_{}.txt".format(self.get_time_string()), "w") as out:
        #         for spec in specs:
        #             out.write("{:.4f}, {}\n".format(spec[0], spec[1]))

        return specs

    def mine_bn_specs(self):
        specs = []
        specs_merge_sets = []
        confusion_matrix = Levenshtein().levenshtein_confusion_matrix(self.paths)

        # perform for every path (or at max N) specification mining iterations
        for i in range(len(confusion_matrix)):
            # print("{}: ".format(i), end='')

            spec = RegexSpec(self.paths[i])

            merge_list = []
            distances = set(confusion_matrix[i])
            for index in [idx for x in [np.where(confusion_matrix[i] == d)[0] for d in distances] for idx in x]:
                merge_list.append(index)
                merge_set = set(merge_list)
                # check for sub-sets
                if any([_set.issubset(merge_set) for _set in specs_merge_sets]):
                    # Rule is redundant
                    break

                Xable = self._get_Xable_flags(index)
                spec.merge(self.paths[index], Xable)

                metrics = self._get_spec_metrics(spec)

                metric_status = self._check_spec_metrics(metrics)

                if metric_status < 0:
                    # one metric exceeded its accepting window, thus dismiss this spec and start
                    # a new iteration with next path
                    break
                elif metric_status > 0:
                    # all metrics lie inside their accepting window, thus we found a spec
                    merge_set = set(merge_list)

                    try:
                        ltl_spec = spec.to_LTL()
                    except:
                        break

                    # check for super-sets => throw them away
                    supersets = [index for index, _set in enumerate(specs_merge_sets) if merge_set.issubset(_set)]
                    if supersets:
                        specs = [s for i, s in enumerate(specs) if i not in supersets]
                        specs_merge_sets = [s for i, s in enumerate(specs_merge_sets) if i not in supersets]
                        # print("Found supersets: {}".format(supersets))

                    # add newly found spec
                    spec_probability = 1 - np.average([p["metric"] for p in [self.bn.paths[i] for i in merge_list]])

                    specs.append((spec_probability, ltl_spec))
                    specs_merge_sets.append(merge_set)

                    break

        specs.sort(key=lambda x: x[0])
        return specs

    @staticmethod
    def is_subset(set1, set2):
        # is set1 a subset of set2
        if len(set1) <= len(set2):
            if all([False for i in set1 if i not in set2]):
                return True

        return False

    @staticmethod
    def _check_spec_metrics(metrics):
        """
        returns 0 if no metric exceeded its accepting window

        return 1 if all metrics lie inside their accepting window

        returns -1 if one exceeded its accepting window
        """
        literal_ratio_window_start = 0.80  # 20% of symbols should be softened
        literal_ratio_window_end = 0.50  # max. 50% of symbols may be softened

        combination_count_window_start = 1 << (int(metrics["length"] / 5) + 1)  # 20% of symbols should be softened
        combination_count_window_end = 1 << (metrics["length"] >> 1)  # max. 50% of symbols may be softened

        all_inside_windows = True

        if metrics["rel_matches"] > literal_ratio_window_start:
            all_inside_windows = False
        elif metrics["rel_matches"] < literal_ratio_window_end:
            return -1

        if metrics["n_matching_strings"] < combination_count_window_start:
            all_inside_windows = False
        elif metrics["n_matching_strings"] > combination_count_window_end:
            return -1

        if all_inside_windows:
            return 1
        else:
            return 0

    def _get_Xable_flags(self, index):
        # returns a list of Xable flags for a given path (passed by the index of the path)
        flags = [False]
        path = self.bn.get_paths()[index]
        for from_node, to_node in zip(path[:-1], path[1:]):
            from_v = from_node.rsplit(":", 1)[0]
            to_v = to_node.rsplit(":", 1)[0]
            if from_v in self.bn.E_histogram:
                if to_v in self.bn.E_histogram[from_v]:
                    flags.append(self.bn.E_histogram[from_v][to_v])
                    continue
            flags.append(False)

        return flags
