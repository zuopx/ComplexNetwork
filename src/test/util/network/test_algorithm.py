import json

import pytest

import src.main.util.network.algorithm as algorithm
import src.main.model.network as network


@pytest.fixture()
def g():
    with open(r'db\DiSFNetwork4\at.json', 'r') as fr:
        at = json.load(fr)
    return network.DirectedGraph('DiSFNetwork4', at)

def test_get_gin_mat_gout_sz(g):
    beta = 0.1
    times = 100
    gin_mat, gout_sz = algorithm.get_gin_mat_gout_szs(g, beta, times)
    print(gin_mat.sum(), gout_sz)
    assert gin_mat.shape == (10000, 100)
    