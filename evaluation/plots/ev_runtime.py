from model_generation.model_generator import ModelGenerator
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

from spec_mining.mining_graph.max_avg_mg import MaxAverageMiningGraph
from spec_mining.mining.comparison_based_miner import CBMiner


def start():
    model_generator = ModelGenerator()

    # Nodes
    model_generator.set_node_range(min_objects=4, max_objects=5,
                                   min_temp_nodes=4, max_temp_nodes=5,
                                   min_states=2, max_states=2)
    # Edges
    model_generator.set_connection_ranges(min_edges_per_object=3, max_edges_per_object=3,
                                          min_percent_inter=0.8, max_percent_inter=0.8)

    rel_remove_list = [r / 10 for r in range(1, 5 + 1, 1)]
    threshold_list = [t / 100 for t in range(80, 100 + 1, 1)]

    n_models = 50

    bn1_paths_list = []

    runtimes = {removed: {th: 0 for th in threshold_list} for removed in rel_remove_list}
    for j in range(n_models):
        print("Model {} of {}".format(j + 1, n_models))

        tscbn1 = model_generator.new_tscbn()
        _start = timer()
        bn1 = MaxAverageMiningGraph(tscbn1)
        setup_1 = timer() - _start

        bn1_paths_list.append(bn1.n_paths)

        for rel_remove in rel_remove_list:
            print("  remove {}".format(rel_remove))
            tscbn2 = model_generator.get_validation_model_rel(tscbn1, rel_remove)
            bn2 = MaxAverageMiningGraph(tscbn2)
            for th in threshold_list:
                _start = timer()
                bn1.path_computation(min_prob_threshold=th)
                if bn1.paths:
                    miner = CBMiner(bn1, bn2)
                    miner.start()
                _end = timer()

                t = _end - _start

                runtimes[rel_remove][th] += (setup_1 + t) / (bn1.n_paths * n_models)

    normalization_factor = np.average(bn1_paths_list)

    for rel_remove, data in runtimes.items():
        y = [runtime * normalization_factor for th, runtime in data.items()]
        plt.plot(threshold_list, y, label=rel_remove)

    plt.legend(loc="upper right")
    plt.legend()
    plt.show()
