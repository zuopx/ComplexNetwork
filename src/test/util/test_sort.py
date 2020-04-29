import json
import functools

import pytest
from scipy import stats

import src.main.util.sort as sort


def test_sort_by_map(nodes, res):
    m = sort.sort_by_map(res)
    nodes_sorted = sorted(nodes, key=lambda x: m(x))
    assert nodes_sorted == [2, 1, 0, 3, 4]


def test_sort_by_cmp():
    folder_nn = 'db/SNAPTwitter/nn/order_independent_net'
    file_emb = 'db/SNAPTwitter/SNAPTwitter.node2vec.emb.json'
    c = sort.sort_by_cmp(folder_nn, file_emb)

    with open('db/SNAPTwitter/seed/20 0.0120 P_GIN.json', 'r') as fr:
        seeds = json.load(fr)

    kts = []
    for nodes in seeds[:20]:
        nodes_nn = sorted(nodes, key=functools.cmp_to_key(c), reverse=True)
        kt = stats.kendalltau(nodes, nodes_nn)
        kts.append(kt)
    print(kts)
    print('kt_avg: ', sum(kts) / len(kts))


def test_sort_by_cmb():
    pass
