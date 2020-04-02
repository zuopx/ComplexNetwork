import pytest
import torch

import src.main.util.nn.training as training
import src.main.model.nn.custom_module as custom_module


def test_train(samples: torch.Tensor):
    conf = training.Conf()
    conf.model = custom_module.TwoLayersSequential
    conf.activation = torch.nn.RReLU
    conf.lr = 1e-2
    D_in = 2
    model = training.train(samples, D_in, conf)
    print(model(samples[:, :D_in]))

