from model_generation.model_generator import ModelGenerator
from spec_mining.mining.comparison_based_miner import CBMiner
from spec_mining.mining_graph.max_avg_mg import MaxAverageMiningGraph
import time
import numpy as np

def start_bayspec():
    print("Hello! Here is BaySpec!")
    print("")
    avgs = []
    for i in range(20):
        model_generator = ModelGenerator()

        # Nodes
        model_generator.set_node_range(min_objects=4, max_objects=4,
                                   min_temp_nodes=5, max_temp_nodes=5,
                                   min_states=2, max_states=2)
        # Edges
        model_generator.set_connection_ranges(min_edges_per_object=2, max_edges_per_object=2,
                                          min_percent_inter=0.8, max_percent_inter=0.8)

        tscbn = model_generator.new_tscbn()

        bn = MaxAverageMiningGraph(tscbn)

        paths = bn.path_computation(min_prob_threshold=0.85)

        if paths:
            validation_bn = MaxAverageMiningGraph(model_generator.get_validation_model_rel(tscbn, 0.25))
            start = time.time()
            miner = CBMiner(bn, validation_bn)
            specs = miner.start()
            end = time.time()
            print("Time %s"% str(end-start));avgs+=[end-start]
            #print("{} found specifications".format(len(specs)))
            #for spec in specs:
            #    print(spec)
    print(np.mean(avgs))