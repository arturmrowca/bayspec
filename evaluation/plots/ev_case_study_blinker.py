from collections import defaultdict

import dill
dill.settings['recurse'] = True
import matplotlib.pyplot as plt

from spec_mining.mining.comparison_based_miner import CBMiner
import evaluation.utility.eval_utility as utility
import evaluation.tools.trace_generation as tg
import evaluation.tools.synoptic as synoptic


def start():
    with open("blinker_model/blinker_tscbn1.tscbn", "rb") as in_stream:
        tscbn1 = dill.load(in_stream)
    with open("blinker_model/blinker_mining_graph_1.tscbn", "rb") as in_stream:
        bn1 = dill.load(in_stream)
    with open("blinker_model/blinker_mining_graph_2.tscbn", "rb") as in_stream:
        bn2 = dill.load(in_stream)

    threshold = 0.8

    bayspec_total = defaultdict(lambda: defaultdict(lambda: 0))
    synoptic_total = defaultdict(lambda: defaultdict(lambda: 0))

    traces_per_log = [t for t in range(2, 5 + 1)]
    samples_per_trace = [s for s in range(2, 5 + 1)]
    log_combinations = len(traces_per_log) * len(samples_per_trace)

    scatter_max_x = []
    scatter_max_y = []

    bn1.path_computation(threshold)
    if bn1.paths:

        miner = CBMiner(bn1, bn2)
        # miner = SingleBnSpecMiner(bn1)
        specs = miner.start()

        if specs:
            bayspec = utility.metrics_from_LTL_specs(specs)

            specs_x = [x for x, ydict in bayspec.items() for y, size in ydict.items()]
            specs_y = [y for x, ydict in bayspec.items() for y, size in ydict.items()]
            specs_s = [size for x, ydict in bayspec.items() for y, size in ydict.items()]

            for h, u, size in zip(specs_x, specs_y, specs_s):
                bayspec_total[h][u] += size

    i = 1
    print("Executing Synoptic and Perracotta:")
    for traces in traces_per_log:
        for samples in samples_per_trace:
            print("Trace set {} / {}".format(i, log_combinations))
            # print("Traces: {}, Samples: {}".format(traces, samples))
            i += 1

            # create Log
            log = tg.single_BN_log(tscbn1, traces_per_log=traces, samples_per_trace=samples)

            # Synoptic
            invariants = synoptic.execute(log, threshold)
            syn = synoptic.metrics_from_invariants(invariants)

            _x = [x for x, ydict in syn.items() for y, size in ydict.items()]
            _y = [y for x, ydict in syn.items() for y, size in ydict.items()]
            _s = [size for x, ydict in syn.items() for y, size in ydict.items()]

            for h, u, size in zip(_x, _y, _s):
                synoptic_total[h][u] += (size / log_combinations)

    # RESULTS
    def process_results(total_dict, histogram, color):
        plot_x = [x for x, ydict in total_dict.items() for y, size in ydict.items()]
        plot_y = [y for x, ydict in total_dict.items() for y, size in ydict.items()]
        plot_s = [size for x, ydict in total_dict.items() for y, size in ydict.items()]

        for height, unique, _s in zip(plot_x, plot_y, plot_s):
            histogram[height] += _s

        if plot_x:
            plt.scatter(x=plot_x, y=plot_y, s=[utility.scatter_size(_s) for _s in plot_s], alpha=0.7, c=color)
            scatter_max_x.append(max(plot_x))
            scatter_max_y.append(max(plot_y))

    # BaySpec
    bayspec_histogram = [0 for _ in range(1000)]
    process_results(bayspec_total, bayspec_histogram, color='#335151')

    # Synoptic
    synoptic_histogram = [0 for _ in range(1000)]
    process_results(synoptic_total, synoptic_histogram, color='#96b7c1')

    max_height = max(scatter_max_x)
    max_y = max(scatter_max_y)
    plt.xlabel("Spec Height")
    plt.ylabel("Unique Events")
    plt.xlim(0, max_height + 1)
    plt.ylim(0, max_y + 1)
    plt.show()

    # BAR PLOT
    # Barplot
    bar_width = 0.9
    bins = list(range(max_height + 1))

    plt.bar(bins, bayspec_histogram[:max_height + 1], color='#335151', width=bar_width, alpha=0.6)
    plt.bar(bins, synoptic_histogram[:max_height + 1], color='#96b7c1', width=bar_width, alpha=0.6)

    plt.xlim(0, max_height + 1)
    plt.xlabel("Spec Height")
    plt.show()
