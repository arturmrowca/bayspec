from collections import defaultdict

from spec_mining.specification.regex_specification import RegexSpec


def metrics_from_LTL_specs(specs):
    M = []
    # put a dict of metrics in M for every spec
    for spec in specs:
        metrics = RegexSpec.evaluation_metrics(spec)
        M.append(metrics)

    metrics = defaultdict(lambda: defaultdict(lambda: 0))

    for spec in M:
        h = spec["height"]
        u = spec["unique"]
        metrics[h][u] += 1

    return metrics


def scatter_size(count):
    size = 100*count
    return size
