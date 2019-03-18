from collections import defaultdict
from numpy import max

INVARIANT_COMPLEXITY = {"precedes": {"h": 2, "w": 3, "u": 2},
                        "follows": {"h": 3, "w": 2, "u": 2},
                        "not_followed": {"h": 4, "w": 2, "u": 2}}


def metrics_from_invariants(invariants):

    d = dict()
    d[INVARIANT_COMPLEXITY["precedes"]["h"]] = dict()
    d[INVARIANT_COMPLEXITY["precedes"]["h"]][INVARIANT_COMPLEXITY["precedes"]["u"]] = len(invariants["precedes"])

    d[INVARIANT_COMPLEXITY["follows"]["h"]] = dict()
    d[INVARIANT_COMPLEXITY["follows"]["h"]][INVARIANT_COMPLEXITY["follows"]["u"]] = len(invariants["follows"])

    d[INVARIANT_COMPLEXITY["not_followed"]["h"]] = dict()
    d[INVARIANT_COMPLEXITY["not_followed"]["h"]][INVARIANT_COMPLEXITY["precedes"]["u"]] = len(invariants["not_followed"])

    return d


def execute(log, th_confidence):
    [counts, precedes, follows] = compute_total_counts(log)

    rel_precedes = []
    rel_follows = []
    not_followed = []

    for first, inner_dict in precedes.items():
        for second, value in inner_dict.items():
            ratio = precedes[first][second] / counts[second]
            if ratio >= th_confidence:
                rel_precedes.append([[first, second], ratio])

    for first, inner_dict in follows.items():
        for second, value in inner_dict.items():
            ratio = follows[first][second] / counts[first]
            if ratio >= th_confidence:
                rel_follows.append([[first, second], ratio])

    for first in counts:
        for second in counts:
            if first != second:
                if follows[first][second] == 0:
                    not_followed.append([first, second])

    rel_precedes.sort(key=lambda x: x[1])
    rel_follows.sort(key=lambda x: x[1])

    return {"precedes": rel_precedes, "follows": rel_follows, "not_followed": not_followed}


def compute_total_counts(log):
    counts = defaultdict(lambda: 0)
    precedes = defaultdict(lambda: defaultdict(lambda: 0))
    follows = defaultdict(lambda: defaultdict(lambda: 0))

    for trace in log:
        [c, p, f] = compute_counts(trace)
        for k, v in c.items():
            counts[k] += v

        for first, inner_dict in p.items():
            for second, value in inner_dict.items():
                precedes[first][second] += value

        for first, inner_dict in f.items():
            for second, value in inner_dict.items():
                follows[first][second] += value

    return [counts, precedes, follows]


def compute_counts(trace):
    indices = defaultdict(lambda: [])
    precedes = defaultdict(lambda: defaultdict(lambda: 0))
    follows = defaultdict(lambda: defaultdict(lambda: 0))

    for i, event in enumerate(trace):
        indices[event].append(i)
        for k in indices:
            if k != event:
                precedes[k][event] += 1

    for i, event in enumerate(trace):
        for k, v in indices.items():
            if k != event:
                if max(v) > i:
                    follows[event][k] += 1

    counts = {k: len(v) for k, v in indices.items()}

    return [counts, precedes, follows]


def false_positive_invariants(invariants):
    total_invariants = 0
    false_positives = 0
    for pattern, invariant_list in invariants.items():
        for invariant in invariant_list:
            total_invariants += 1
            if is_false_positive(invariant[0][0], invariant[0][1]):
                false_positives += 1

    return [false_positives, total_invariants]


def is_false_positive(a, b):
    f1 = a[:2]
    f2 = b[:2]
    if f1 != f2:
        return True

    return False
