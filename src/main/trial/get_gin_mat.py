import json

from scipy import sparse

import src.main.util.network.algorithm as algorithm
import src.main.model.network as network

g_name = 'DiSFNetwork20'
at_path = f'db/{g_name}/at.json'

with open(at_path, 'r')  as fr:
    at = json.load(fr)

g = network.DirectedGraph(g_name, at)

betas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

for beta in betas:
    print(f'{beta:.4f}')

    gin_mat, gout_szs = algorithm.get_n_gin_mat_gout_szs(g, beta)
    
    gin_mat_path = f'db/{g_name}/gin_mat/n{beta:.4f}.npz'
    sparse.save_npz(gin_mat_path, gin_mat)

    gout_szs_path = f'db/{g_name}/gout_sz/n{beta:.4f}.json'
    with open(gout_szs_path, 'w') as fw:
        json.dump(gout_szs_path, fw)
