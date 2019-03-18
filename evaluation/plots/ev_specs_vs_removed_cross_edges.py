from model_generation.model_generator import ModelGenerator
from spec_mining.mining.comparison_based_miner import CBMiner
from spec_mining.mining_graph.max_avg_mg import MaxAverageMiningGraph

import network.utility as model_utility

import copy
import matplotlib.pyplot as plt


def start():
    max_remove = 75
    rel_remove = list(range(max_remove + 1))
    thresholds = [th / 100 for th in [84, 86, 88, 90]]
    n_models = 50

    model_generator = ModelGenerator()

    # Nodes
    model_generator.set_node_range(min_objects=4, max_objects=5,
                                   min_temp_nodes=4, max_temp_nodes=5,
                                   min_states=2, max_states=2)
    # Edges
    model_generator.set_connection_ranges(min_edges_per_object=3, max_edges_per_object=3,
                                          min_percent_inter=0.8, max_percent_inter=0.8)

    cummulative_paths = {th: [0 for _ in rel_remove] for th in thresholds}
    cummulative_specs = {th: [0 for _ in rel_remove] for th in thresholds}

    for i in range(n_models):
        print("Model {} of {}".format(i + 1, n_models))
        tscbn1 = model_generator.new_tscbn()

        mining_graph = MaxAverageMiningGraph(tscbn1)
        n_cross_edges = model_utility.get_n_cross_edges(tscbn1)

        all_paths = mining_graph.path_computation(thresholds[0])

        for th in thresholds:
            paths = [p for p in all_paths if p["metric"] <= (1 - th)]
            mining_graph.paths = paths

            print("  p_min={} ({} paths)".format(th, len(mining_graph.paths)))

            if mining_graph.paths:

                tscbn2 = copy.deepcopy(tscbn1)
                last_specs = 0

                for r in rel_remove:
                    # if round(remove) > last_removed:
                    if r == 0 or round(n_cross_edges * r / 100) > round(n_cross_edges * (r - 1) / 100):
                        delta = round(n_cross_edges * r / 100) - round(n_cross_edges * (r - 1) / 100)
                        tscbn2 = model_generator.get_validation_model_abs(tscbn2, abs_remove=delta)

                        validation_graph = MaxAverageMiningGraph(tscbn2)
                        miner = CBMiner(mining_graph, validation_graph)
                        specs = miner.start()

                        cummulative_paths[th][r] += len(mining_graph.paths)
                        cummulative_specs[th][r] += len(specs)
                        last_specs = len(specs)

                    else:
                        cummulative_paths[th][r] += len(mining_graph.paths)
                        cummulative_specs[th][r] += last_specs

                    # print("remove: {}% ({}) - specs: {}".format(r, round(n_cross_edges * r / 100),last_specs))

    data = {th: [cummulative_specs[th][r] / n_models for r in rel_remove] for th in thresholds}

    for threshold, ys in data.items():
        plt.plot(rel_remove, ys, label=threshold)

    plt.legend(loc="lower left")
    plt.legend()
    plt.grid()
    plt.show()
    plt.show()
