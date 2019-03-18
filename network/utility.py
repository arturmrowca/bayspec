def get_n_cross_edges(tscbn):
    cross_edges = [1 for e in tscbn.E if e[0].rsplit("_", 1)[0] != e[1].rsplit("_", 1)[0] and not (
            str.startswith(e[0], "dL_") or str.startswith(e[1], "dL_"))]

    return len(cross_edges)