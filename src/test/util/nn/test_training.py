import pytest
import torch
import matplotlib.pyplot as plt

import src.main.util.nn.training as training
import src.main.model.nn.custom_module as custom_module

import config

DEVICE = config.get_DEVICE()

def test_classify(samples: torch.Tensor):
    conf = training.Conf()
    conf.model = custom_module.OrderIndependentNet
    conf.activation = torch.nn.RReLU
    conf.loss_fn = torch.nn.NLLLoss()
    conf.lr = 1e-2
    D_out = 2
    model = training.classify(samples, D_out, conf)
    print((model(samples[:, :-1].to(device=DEVICE))))

def test_hit_rate(class_y_guess, class_y):
    assert training.hit_rate(class_y_guess, class_y) == 0
