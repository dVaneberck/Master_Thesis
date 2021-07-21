import minerl
import math
import random
import gym as gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from IPython.core.display import clear_output
import torchvision.transforms as T
import datetime
import argparse
from pathlib import Path
import time

from prioritized__experience_replay import *
from neural_net import *
from Logger import *


class Agent:
    # abstract class

    def __init__(self, network, nb_actions):

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.EPISODES = 400
        self.memory = deque(maxlen=2000)
        self.per_memory = PrioritizedExperienceReplay(10000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.0005
        self.current_step = 0

        self.batch_size = 32
        self.nFrames = 2
        self.train_start = 1000
        self.target_sync = 2  # in episodes

        self.ddqn = True
        self.epsilon_greedy = True
        self.PER_use = True

        # ! change neural net according to args.network
        self.model = network(input_shape=self.nFrames, action_space=nb_actions).float()
        self.model = self.model.to(self.device)

        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.loss_fn = torch.nn.SmoothL1Loss().to(device=self.device)

        save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        self.logger = MetricLogger(save_dir)

    def update_target(self):
        if self.ddqn:
            self.model.target.load_state_dict(self.model.online.state_dict())

    def store(self, state, action, reward, next_state, done):
        if self.device.type == 'cuda':
            state = torch.tensor(state, device=self.device)
            next_state = torch.tensor(next_state, device=self.device)
            action = torch.tensor([action], device=self.device)
            reward = torch.tensor([reward], device=self.device)
            done = torch.tensor([done], device=self.device)
        else:
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        if self.PER_use:
            self.per_memory.push((state, next_state, action, reward, done))
        else:
            self.memory.append((state, next_state, action, reward, done))

    def act(self, state, decay_step):
        pass

    def update_net(self, reward_log):
        pass

