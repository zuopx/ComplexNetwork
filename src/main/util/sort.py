"""只解决一个问题：给定候选点集，给它们排序。"""
import os
import json

import torch

import config
import src.main.model.nn.custom_module as custom_module

DEVICE = config.get_DEVICE()


def sort_by_map(file):  # map, return function
    """
    file -- path of a .json file containing a list
    """
    with open(file, 'r') as fr:
        m = json.load(fr)

    def _sort_by_map(x: int):
        return m[x]

    return _sort_by_map


def sort_by_cmp(folder_nn, file_emb):
    """
    folder -- path of a folder containing some nn models
    file -- path of node embedding containing a dict
    """
    models = []
    for file in os.listdir(folder_nn):
        model = custom_module.OrderIndependentNet(256, 2, torch.nn.RReLU)
        model.load_state_dict(torch.load(
            os.path.join(folder_nn, file), map_location=DEVICE))
        model.eval()
        models.append(model)

    with open(file_emb, 'r') as f:
        emb = json.load(f)

    def _sort_by_cmp(x1: int, x2: int):
        x = torch.Tensor(emb[str(x1)] + emb[str(x2)]).unsqueeze(0)
        s = 0  # x1x2为逆序的次数
        for model in models:
            y = model(x)
            # c: class, 顺序，属于类别0；逆序，属于类别1.
            c = y.max(axis=1).indices.item()
            s += c
        return s - len(models) / 2

    return _sort_by_cmp


def sort_by_cmb(nodes: list, ) -> list:  # combination
    pass
