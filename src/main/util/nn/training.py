import torch
import matplotlib.pyplot as plt
import matplotlib

import src.main.model.nn.custom_module as custom_module
import src.main.util.nn.sample as sample

import config

DEVICE = config.get_DEVICE()


class Conf:
    def __init__(self):
        self.model = custom_module.ThereLayersSequential
        self.activation = torch.nn.RReLU
        self.optimizer = torch.optim.Adam
        self.loss_fn = torch.nn.MSELoss()
        self.lr = 1e-2
        self.weight_decay = 0.0
        self.n_epochs = 1000
        self.early_stop_epoch = None

    def __repr__(self):
        return ('Conf:\n' +
                f'model: {self.model}\n' +
                f'activation: {self.activation}\n' +
                f'optimizer: {self.optimizer}\n' +
                f'loss_fn: {self.loss_fn}\n' +
                f'lr: {self.lr}\n' +
                f'weight_decay: {self.weight_decay}\n' +
                f'n_epochs: {self.n_epochs}\n' +
                f'early_stop_epoch: {self.early_stop_epoch}')


def classify(samples: torch.Tensor, D_out: int, conf: Conf, ax=None):
    # assert isinstance(conf.loss_fn, torch.nn.CrossEntropyLoss)
    D_in = samples.size()[1] - 1
    training_set, validation_set = sample.split_samples(samples)
    training_x = training_set[:, :D_in].to(device=DEVICE)
    validation_x = validation_set[:, :D_in].to(device=DEVICE)
    training_y = training_set[:, D_in:].squeeze(1).long().to(device=DEVICE)
    validation_y = validation_set[:, D_in:].squeeze(1).long().to(device=DEVICE)
    training_losses, validation_losses = [], []
    training_hr, validation_hr = [], []
    model = conf.model(D_in, D_out, conf.activation).to(device=DEVICE)
    optimizer = conf.optimizer(
        model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    for epoch in range(conf.n_epochs):
        training_y_guess = model(training_x)
        validation_y_guess = model(validation_x)
        training_loss = conf.loss_fn(training_y_guess, training_y)
        validation_loss = conf.loss_fn(validation_y_guess, validation_y)
        training_losses.append(training_loss.item())
        validation_losses.append(validation_loss.item())

        training_hit_rate = hit_rate(training_y_guess, training_y)
        validation_hit_rate = hit_rate(validation_y_guess, validation_y)
        training_hr.append(training_hit_rate)
        validation_hr.append(validation_hit_rate)

        if conf.early_stop_epoch and early_stop(validation_losses, conf.early_stop_epoch):
            break
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()
    if DEVICE == torch.device("cuda:0"):
        torch.cuda.empty_cache()
    training_hit_rate = hit_rate(training_y_guess, training_y)
    validation_hit_rate = hit_rate(validation_y_guess, validation_y)
    print(f'training hit rate --> {training_hit_rate: .8f},\
          validation hit rate --> {validation_hit_rate: .8f}')
    if ax:
        plot_loss(ax, training_losses, validation_losses)
    return model, training_losses, validation_losses, training_hr, validation_hr


def early_stop(validation_losses: list, epoch: int) -> bool:
    if len(validation_losses) >= epoch:
        if torch.all(validation_losses[-epoch] <=
                     torch.Tensor(validation_losses[-epoch:])):
            return True
    return False


def hit_rate(y_guess: torch.Tensor, y: torch.Tensor):
    return y_guess.max(axis=1).indices.eq(y).sum().item() / y.size()[0]


def regress(samples: torch.Tensor, D_out: int, conf: Conf, ax=None):
    D_in = samples.size()[1] - 1
    training_set, validation_set = sample.split_samples(samples)
    training_x = training_set[:, :D_in].to(device=DEVICE)
    training_y = training_set[:, D_in:].to(device=DEVICE)
    validation_x = validation_set[:, :D_in].to(device=DEVICE)
    validation_y = validation_set[:, D_in:].to(device=DEVICE)
    training_losses, validation_losses = [], []
    model = conf.model(D_in, D_out, conf.activation).to(device=DEVICE)
    optimizer = conf.optimizer(model.parameters(), lr=conf.lr)
    for epoch in range(conf.n_epochs):
        training_loss = conf.loss_fn(model(training_x), training_y)
        validation_loss = conf.loss_fn(model(validation_x), validation_y)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()
    if ax:
        plot_loss(ax, training_losses, validation_losses)
    else:
        print(
            f'training loss --> {training_losses[-1]:.8f}, validation loss --> {validation_losses[-1]:.8f}')
    return model


def plot_loss(ax: matplotlib.axes.Axes, training_losses: list, validation_losses: list):
    sz = len(training_losses)
    ax.plot(range(sz), training_losses, c='r',
            label='t--' + str(training_losses[-1]))
    ax.plot(range(sz), validation_losses, c='g',
            label='v--' + str(validation_losses[-1]))
    # ax.legend(fontsize=16)
    ax.grid(alpha=0.25)
