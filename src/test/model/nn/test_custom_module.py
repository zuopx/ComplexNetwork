import pytest
import torch

import src.main.model.nn.custom_module as custom_module


def show_modules(modules):
    for module in modules:
        print(module)


def test_there_layers_sequential():
    s = custom_module.ThereLayersSequential(10, 1, torch.nn.ReLU)
    show_modules(s.modules())
    # print("Done!")


def test_order_independent_net_init():
    net = custom_module.OrderIndependentNet(10, 5, torch.nn.Tanh)
    show_modules(net.modules())


def test_order_independent_net_forword():
    D_in = 100
    net = custom_module.OrderIndependentNet(D_in, 10, torch.nn.Tanh)
    x = torch.rand(2, D_in)
    y = net(x)
    print(y)

