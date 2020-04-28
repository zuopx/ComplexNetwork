"""process samples"""
import torch
import random


def get_samples_sequential(nodes: list, node_emb: dict, target: list,
                            num_samples: int = 300) -> torch.Tensor:
    sample_nodes = random.sample(nodes, num_samples)
    samples = torch.Tensor([node_emb[str(node)] + [target[node]]
                            for node in sample_nodes])
    return samples


def get_samples_order_independent_net(nodes: list, node_emb: dict,
                                      target: list, num_samples: int = 40000) -> torch.Tensor:
    num_nodes = len(nodes)
    assert num_samples <= num_nodes * (num_nodes - 1) / 2
    samples = []
    for _ in range(num_samples):
        p = random.sample(nodes, 2)
        sample = node_emb[str(p[0])] + node_emb[str(p[1])] + \
            [int(target[p[0]] > target[p[1]])]
        samples.append(sample)
    samples = torch.Tensor(samples)
    return samples


def split_samples(samples: torch.Tensor, training_rate: float = 0.2) -> list:
    num_samples = samples.size()[0]
    permutation = list(range(num_samples))
    random.shuffle(permutation)
    num_training = round(num_samples * (1 - training_rate))
    training_set = samples[permutation[:num_training]]
    validation_set = samples[permutation[num_training:]]
    return training_set, validation_set
