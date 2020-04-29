import networkx as nx


def scale_free_directed_graph(n: int, alpha=0.41, beta=0.54, gamma=0.05,
                              delta_in=0.2, delta_out=0) -> list:
    assert alpha + beta + gamma == 1.0
    g = nx.scale_free_graph(n, alpha, beta, gamma, delta_in, delta_out)
    at = [list(g[i]) for i in range(n)]
    return at

def status(at: list):
    N = len(at)
    out_degree = [len(_) for _ in at]
    in_degree = [0 for _ in range(N)]
    for i in range(N):
        for j in at[i]:
            in_degree[j] += 1
    print('\t', r'$k^{out}_{max}$: ', max(out_degree))
    print('\t', r'$p_o$: ', out_degree.count(0) / N)
    print('\t', r'$p_i$: ', in_degree.count(0) / N)
