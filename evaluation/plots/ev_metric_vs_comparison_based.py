from model_generation.model_generator import ModelGenerator

import matplotlib.pyplot as plt

from spec_mining.mining_graph.max_avg_mg import MaxAverageMiningGraph
from spec_mining.mining.comparison_based_miner import CBMiner
from spec_mining.mining.metric_based_miner import MBMiner


def start():
    n_models = 2
    objects = [4, 5]
    nodes_per_object = [4, 5]
    states_per_node = 2
    edges_per_object = 3
    percentage_inter = 0.8

    edge_remove_ratio = 0.20
    min_th = 80
    thresholds = [t / 100 for t in range(min_th, 100 + 1)]

    n_eval_models = 50
    sizes = ["(4,4)", "(5,5)"]

    MG_list = []
    for i in range(n_models):
        model_generator = ModelGenerator()

        # Nodes
        model_generator.set_node_range(min_objects=objects[i], max_objects=objects[i],
                                       min_temp_nodes=nodes_per_object[i], max_temp_nodes=nodes_per_object[i],
                                       min_states=states_per_node, max_states=states_per_node)
        # Edges
        model_generator.set_connection_ranges(min_edges_per_object=edges_per_object, max_edges_per_object=edges_per_object,
                                              min_percent_inter=percentage_inter, max_percent_inter=percentage_inter)

        MG_list.append(model_generator)

    mb_cummulative = {size: {th: 0 for th in thresholds} for size in range(n_models)}
    cb_cummulative = {size: {th: 0 for th in thresholds} for size in range(n_models)}

    for j in range(n_eval_models):
        print("Model {} of {}".format(j + 1, n_eval_models))
        for i in range(n_models):
            print("  size:{}".format(sizes[i]))
            tscbn1 = MG_list[i].new_tscbn()
            bn1 = MaxAverageMiningGraph(tscbn1)
            tscbn2 = MG_list[i].get_validation_model_rel(tscbn1, edge_remove_ratio)
            bn2 = MaxAverageMiningGraph(tscbn2)

            # all paths for the minimal threshold
            all_paths = bn1.path_computation(min_th / 100)

            for th in thresholds:
                print("    threshold:{}".format(th))
                paths = [p for p in all_paths if p["metric"] <= (1 - th)]
                bn1.paths = paths

                metric_based_miner = MBMiner(bn1)
                comparison_based_miner = CBMiner(bn1, bn2)

                mb_specs = metric_based_miner.start()
                cb_specs = comparison_based_miner.start()

                mb_cummulative[i][th] += len(mb_specs)
                cb_cummulative[i][th] += len(cb_specs)

    colors = ["#0065BD", "#E37222"]
    for i in range(n_models):
        mb_data = [mb_cummulative[i][th] / n_eval_models for th in thresholds]
        plt.plot(thresholds, mb_data, label="mb {}".format(sizes[i]), linestyle="dashed", color=colors[i])

        cb_data = [cb_cummulative[i][th] / n_eval_models for th in thresholds]
        plt.plot(thresholds, cb_data, label="cb {}".format(sizes[i]), linestyle="solid", color=colors[i])

    plt.legend()
    plt.show()
