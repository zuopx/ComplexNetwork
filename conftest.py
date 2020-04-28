import pytest
import json
import os
import config
import torch

DB = config.get_DB()


@pytest.fixture()
def SNAPTwitter_nodes() -> list:
    file = os.path.join(DB, 'SNAPTwitter/nodes_degree_approx_20.json')
    with open(file, 'r') as f:
        nodes = json.load(f)
    return nodes


@pytest.fixture()
def SNAPTwitter_emb() -> dict:
    file = os.path.join(DB, 'SNAPTwitter/SNAPTwitter.node2vec.emb.json')
    with open(file, 'r') as f:
        node_emb = json.load(f)
    return node_emb


@pytest.fixture()
def SNAPTwitter_p_gins() -> list:
    file = os.path.join(DB, 'SNAPTwitter/p_gin_beta0.0120.json')
    with open(file, 'r') as f:
        p_gins = json.load(f)
    return p_gins


@pytest.fixture()
def samples() -> torch.Tensor:
    samples = [[i, i ** 2, i % 2] for i in range(10)]
    return torch.Tensor(samples)


@pytest.fixture()
def class_y_guess():
    y = [[i, 10 - i] for i in range(10)]
    return torch.Tensor(y)


@pytest.fixture()
def class_y():
    return torch.Tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])


@pytest.fixture()
def validation_losses():
    return list(range(9)) + [0]


@pytest.fixture()
def nodes():
    return list(range(5))


@pytest.fixture()
def res():
    return 'src/test/util/nodes_map.json'