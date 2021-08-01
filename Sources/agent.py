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
import torch
import torch.nn as nn
import torch.optim as optim

from prioritized__experience_replay import *
from neural_net import *
from Logger import *


class Agent:
    # abstract class

    def __init__(self, network, n_features, nb_actions):

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.EPISODES = 400
        self.memory = deque(maxlen=2000)
        self.per_memory = PrioritizedExperienceReplay(10000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.0005

        self.batch_size = 32
        self.nFrames = 2
        self.train_start = 1000
        self.target_sync = 2  # in episodes

        self.ddqn = True
        self.epsilon_greedy = True
        self.PER_use = True

        # ! change neural net according to args.network
        self.model = network(input_shape=self.nFrames, action_space=nb_actions, n_features=n_features).float()
        self.model = self.model.to(self.device)

        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.loss_fn = torch.nn.SmoothL1Loss().to(device=self.device)

        save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        self.logger = MetricLogger(save_dir)

    def reset(self):
        pass

    def act(self, state, decay_step):
        pass

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

    def update_net(self, reward_log):
        if self.PER_use:
            minibatch, tree_idx = self.per_memory.sample(self.batch_size)
        else:
            if len(self.memory) < self.batch_size:
                return
            minibatch = random.sample(self.memory, self.batch_size)

        state, next_state, action, reward, done = map(torch.stack, zip(*minibatch))

        action = action.squeeze()
        reward = reward.squeeze()
        done = done.squeeze()

        online = self.model(state, model='online')  # all Q-values predicted
        ab = np.arange(0, self.batch_size)
        online_Q = online[  # Q-value predicted for the chosen action
            ab, action
        ]

        if self.ddqn:
            target_next = self.model(next_state, model='online')  # predicted Q-values for next state
            best_action = torch.argmax(target_next, axis=1)  # best action according to target_next

            next_Q = self.model(next_state, model='target')[  # target Q-value for best action at next state
                ab, best_action
            ]
            td = (reward + (1 - done.float()) * self.gamma * next_Q).float()

        else:
            target_next = self.model(state, model='online')
            best_action = torch.argmax(target_next, axis=1)

            next_Q = self.model(state, model='online')[
                ab, best_action
            ]
            td = (reward + (1 - done.float()) * self.gamma * next_Q).float()

        loss = self.loss_fn(online_Q, td)

        if self.PER_use:
            absolute_errors = (online_Q - next_Q).abs()
            # Update priority
            self.per_memory.update_priorities(tree_idx, absolute_errors)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        test = reward_log
        self.logger.log_step(reward=test, loss=loss.item(), q=online_Q.mean().item())


