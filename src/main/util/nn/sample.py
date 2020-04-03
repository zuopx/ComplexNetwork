"""process samples"""
import torch
import random


def get_samples_sequential():
    pass


def get_samples_order_independent_net(nodes: list, node_emb: dict,
                                      target: list, sample_rate: float=0.1) -> torch.Tensor:
    samples_num = round(len(nodes) * sample_rate)
    sample_nodes = random.sample(nodes, samples_num)
    pairs = [(sample_nodes[i], sample_nodes[j])
             for i in range(samples_num) for j in range(i + 1, samples_num)]
    samples = []
    for p in pairs:
        sample = node_emb[str(p[0])] + node_emb[str(p[1])] + [int(target[p[0]] > target[p[1]])]
        samples.append(sample)
    samples = torch.Tensor(samples)
    return samples


def split_samples(samples: torch.Tensor, training_rate: float=0.2) -> list:
    samples_num = samples.size()[0]
    permutation = list(range(samples_num))
    random.shuffle(permutation)
    training_num = round(samples_num * (1 - training_rate))
    training_set = samples[permutation[:training_num]]
    validation_set = samples[permutation[training_num:]]
    return training_set, validation_set