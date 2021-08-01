import torch
import torch.nn as nn
import copy


class NeuralNet(nn.Module):
    # interface

    def forward(self, x, model):
        pass


class SmallMLP(NeuralNet):

    def __init__(self, input_shape, action_space, **kwargs):
        super(SmallMLP, self).__init__()
        self.online = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model):
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)


class BigMLP(NeuralNet):

    def __init__(self, input_shape, action_space, **kwargs):
        super(BigMLP, self).__init__()
        self.online = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model):
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)


class ConvNet(NeuralNet):

    def __init__(self, input_shape, action_space, **kwargs):  # changer parametres et prendre input size
        super(ConvNet, self).__init__()
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(kwargs['n_features'], 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model):
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)
