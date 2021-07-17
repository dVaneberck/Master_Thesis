import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy


class NeuralNet(nn.Module):

    def __init__(self, input_shape, action_space):
        super(NeuralNet, self).__init__()
        # self.online = nn.Sequential(
        #     nn.Linear(input_shape, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, action_space),
        # )

        self.online = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(), # sigmoid ?
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
