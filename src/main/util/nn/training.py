import torch

import src.main.model.nn.custom_module as custom_module
import src.main.util.nn.sample as sample

class Conf:
    def __init__(self):
        self.model= custom_module.ThereLayersSequential
        self.activation= torch.nn.Tanh
        self.optimizer= torch.optim.Adam
        self.loss_fn = torch.nn.MSELoss()
        self.lr = 1e-4
        self.n_epochs = 1000

def train(samples: torch.Tensor, D_in: int, conf: Conf):
    x = samples[:, :D_in]
    y = samples[:, D_in:]
    model = conf.model(D_in, y.size()[1], conf.activation)
    optimizer = conf.optimizer(model.parameters(), lr=conf.lr)
    for epoch in range(conf.n_epochs):
        y_g = model(x)
        loss = conf.loss_fn(y, y_g)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


