import pytest
import json
import os
import config
import torch

DB = config.get_DB()


@pytest.fixture()
def SNAPTwitter_nodes() -> list:
    file = os.path.join(DB, 'snap_twitter/nodes_degree_approx_20.json')
    with open(file, 'r') as f:
        nodes = json.load(f)
    return nodes


@pytest.fixture()
def SNAPTwitter_emb() -> dict:
    file = os.path.join(DB, 'snap_twitter/snap_twitter.node2vec.emb.json')
    with open(file, 'r') as f:
        node_emb = json.load(f)
    return node_emb


@pytest.fixture()
def SNAPTwitter_p_gins() -> list:
    file = os.path.join(DB, 'snap_twitter/p_gin_beta0.0120.json')
    with open(file, 'r') as f:
        p_gins = json.load(f)
    return p_gins


@pytest.fixture()
def samples() -> torch.Tensor:
    samples = [[i, i**2, i**3] for i in range(10)]
    return torch.Tensor(samples)
