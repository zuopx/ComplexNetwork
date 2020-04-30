"""Get out_degree, k_shell, ci, betweenness, closeness"""
import os
import json

import config
import src.main.util.network.algorithm as algorithm
import src.main.model.network as network

DB = config.get_DB()

g_name = 'DiSFNetwork4'

base_path = os.path.join(DB, g_name)
at_path = os.path.join(base_path, 'at.json') 

with open(at_path, 'r') as fr:
    at = json.load(fr)

def main():
    out_degree = [len(_) for _ in at]

    k_shell = algorithm.Algorithm.get_k_shell_for_undirected_graph(at)

    betweenness = algorithm.Algorithm.get_betweenness(at)

    closeness = []
    for i in range(len(at)):
        closeness.append(algorithm.Algorithm.get_closeness(at, i))
    
    ci = []
    for i in range(len(at)):
        ci.append(algorithm.Algorithm.get_collective_influence(at, i))

    data = {'out_degree': out_degree, 
            'k_shell': k_shell,
            'betweenness': betweenness,
            'closeness': closeness, 
            'ci': ci}
    for k in data.keys():
        with open(os.path.join(base_path, k), 'r') as fr:
            json.load(fr)


if __name__ == "__main__":
    main()
