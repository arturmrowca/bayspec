import random
import matplotlib.pyplot as plt

from spec_mining.mining.comparison_based_miner import CBMiner
from model_generation.model_generator import ModelGenerator
from spec_mining.mining_graph.max_avg_mg import MaxAverageMiningGraph

import evaluation.tools.synoptic as synoptic
import evaluation.tools.trace_generation as tg


def start():
    model_generator = ModelGenerator()

    # Nodes
    model_generator.set_node_range(min_objects=4, max_objects=5,
                                   min_temp_nodes=4, max_temp_nodes=5,
                                   min_states=2, max_states=2)
    # Edges
    model_generator.set_connection_ranges(min_edges_per_object=3, max_edges_per_object=3,
                                          min_percent_inter=0.8, max_percent_inter=0.8)

    threshold = 0.8
    edge_remove_ratio = 0.3

    model_pool_size = 50
    model_pool = []
    bayspec_fps_per_model = [[0,0] for _ in range(model_pool_size)]
    for i in range(model_pool_size):
        print("generate network {} of {}".format(i+1,model_pool_size))
        tscbn1 = model_generator.new_tscbn()
        model_pool.append(tscbn1)

        tscbn2 = model_generator.get_validation_model_rel(tscbn1, edge_remove_ratio)

        bn1 = MaxAverageMiningGraph(tscbn1)
        bn2 = MaxAverageMiningGraph(tscbn2)

        bn1.path_computation(threshold)

        miner = CBMiner(bn1, bn2)

        fps, specs = miner.start(evaluation=True)

        bayspec_fps_per_model[i] = [fps, len(specs)]

    min_models = 2
    max_models = 5
    n_models_list = list(range(min_models, max_models + 1))
    n_loops = 10

    traces_per_log = list(range(2, 5 + 1))
    samples_per_trace = list(range(2, 5 + 1))
    log_combinations = len(traces_per_log) * len(samples_per_trace)

    synoptic_fps = {n: [0, 0] for n in n_models_list}
    bayspec_fps = {n: [0, 0] for n in n_models_list}

    for n_models in n_models_list:
        print("n_models: {}".format(n_models))
        for loop in range(n_loops):
            print("  loop: {} / {}".format(loop+1,n_loops))
            index_tscbns = random.sample(list(enumerate(model_pool)), n_models)
            indices = [index for index, tscbn in index_tscbns]
            tscbn_subset = [tscbn for index, tscbn in index_tscbns]

            for index in indices:
                bayspec_fps[n_models][0] += (bayspec_fps_per_model[index][0] / n_loops)
                bayspec_fps[n_models][1] += (bayspec_fps_per_model[index][1] / n_loops)

            for traces in traces_per_log:
                for samples in samples_per_trace:
                    log = tg.multi_BN_log_interleaving(tscbn_subset, traces, samples)

                    # Synoptic
                    invariants = synoptic.execute(log, threshold)
                    syn_fp = synoptic.false_positive_invariants(invariants)
                    if syn_fp[1] > 0:
                        synoptic_fps[n_models][0] += (syn_fp[0] / (n_loops * log_combinations))
                        synoptic_fps[n_models][1] += (syn_fp[1] / (n_loops * log_combinations))

    bar_width = 0.9
    for i, n_nets in enumerate(n_models_list):
        if synoptic_fps[n_nets][1] > 0:
            plt.bar([2 * i + 0], [100 * synoptic_fps[n_nets][0] / synoptic_fps[n_nets][1]], color='#96b7c1',
                    width=bar_width)
        if bayspec_fps[n_nets][1] > 0:
            plt.bar([2 * i + 1], [100 * bayspec_fps[n_nets][0] / bayspec_fps[n_nets][1]], color='#335151',
                    width=bar_width)

    plt.show()
