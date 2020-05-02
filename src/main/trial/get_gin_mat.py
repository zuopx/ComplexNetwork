import json

import numpy as np
from scipy import sparse

import src.main.util.network.algorithm as algorithm
import src.main.model.network as network

g_name = 'DiSF_b'
at_path = f'db/{g_name}/at.json'

with open(at_path, 'r')  as fr:
    at = json.load(fr)

g = network.DirectedGraph(g_name, at)

mode = 'nsir'

# betas = (np.arange(40) + 1) *  0.0100  # DiSF_a
betas = (np.arange(40) + 1) * (0.0800 / 40)  # DiSF_b
 
def main():
    for beta in betas:
        print(f'{beta:.4f}')

        if mode == 'nsir':
            gin_mat, gout_szs = algorithm.get_n_gin_mat_gout_szs(g, beta)
            
            gin_mat_path = f'db/{g_name}/gin_mat/n{beta:.4f}.npz'
            sparse.save_npz(gin_mat_path, gin_mat)

            gout_szs_path = f'db/{g_name}/gout_sz/n{beta:.4f}.json'
            with open(gout_szs_path, 'w') as fw:
                json.dump(gout_szs, fw)
        elif mode == 'sir':
            gin_mat, gout_szs = algorithm.get_gin_mat_gout_szs(g, beta)
            
            gin_mat_path = f'db/{g_name}/gin_mat/{beta:.4f}.npz'
            sparse.save_npz(gin_mat_path, gin_mat)

            gout_szs_path = f'db/{g_name}/gout_sz/{beta:.4f}.json'
            with open(gout_szs_path, 'w') as fw:
                json.dump(gout_szs, fw)

if __name__ == "__main__":
    print(g_name, mode, betas)
    main()