import torch


class TwoLayersSequential(torch.nn.Sequential):
    def __init__(self, D_in: int, D_out: int, activation: type):
        super().__init__()
        h = D_in - (D_in - D_out) // 2
        self.linear1 = torch.nn.Linear(D_in, h)
        self.activation1 = activation()
        self.linear2 = torch.nn.Linear(h, D_out)
        self.activation2 = activation()


class ThereLayersSequential(torch.nn.Sequential):
    def __init__(self, D_in: int, D_out: int, activation: type=torch.nn.Tanh):
        super().__init__()
        h1 = D_in - (D_in - D_out) // 3
        h2 = D_in - (D_in - D_out) // 3 * 2
        self.linear1 = torch.nn.Linear(D_in, h1)
        self.activation1 = activation()
        self.linear2 = torch.nn.Linear(h1, h2)
        self.activation2 = activation()
        self.linear3 = torch.nn.Linear(h2, D_out)
        self.activation3 = activation()


class OrderIndependentNet(torch.nn.Module):
    def __init__(self, D_in: int, D_out: int, activation: type):
        super().__init__()
        self._D_out = D_out
        self._sequential_d_in = D_in // D_out
        assert D_in % D_out == 0, f"{D_in} must be divisible by {D_out}."
        self.sequential = ThereLayersSequential(D_in // D_out, 1, activation)

    def forward(self, x):
        y = torch.empty(x.size()[0], self._D_out)
        for i in range(self._D_out):
            y[:, i] = self.sequential(
                x[:, i * self._sequential_d_in:(i + 1) * self._sequential_d_in]).squeeze()
        return y
