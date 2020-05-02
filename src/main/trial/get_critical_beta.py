import json

import numpy as np

import src.main.model.network as network
import src.main.util.network.algorithm as algorithm


g_name, times = 'DiSF_b', 100

betas = (np.arange(100) + 1) * (0.02 / 100)  # DiSF_b

# betas = (np.arange(100) + 1) * (0.015 / 100)  # Twitter
# betas = (np.arange(100) + 1) * (0.015 / 100)  # Epinions


def main():
    # g_name, betas, times
    with open(f'db/{g_name}/at.json', 'r') as fr:
        at = json.load(fr)
    g = network.DirectedGraph(g_name, at)

    gout_mean, gout_std = [], []
    for beta in betas:
        print(f'{beta:.4f},', end='\t')
        gout_szs = []
        for t in range(times):
            sub_at = g.n_bond_percolation(beta)
            gout = algorithm.Algorithm.get_gout(sub_at)
            gout_szs.append(len(gout))
        gout_szs = np.array(gout_szs)
        gout_mean.append(gout_szs.mean())
        gout_std.append(gout_szs.std())

    x = [float(_) for _ in betas]
    y1 = [float(_) for _ in np.array(gout_mean) / g.node_num]
    y2 = [float(_) for _ in np.array(gout_std) / np.array(gout_mean)]
    data = {
        'x': x,
        'y1': y1,
        'y2': y2
    }
    with open(f'db/{g_name}/img/critical_beta.json', 'w') as fw:
        json.dump(data, fw)


if __name__ == "__main__":
    print(g_name, betas)
    main()
