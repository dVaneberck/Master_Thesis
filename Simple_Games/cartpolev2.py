import math
import random

import gym as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from IPython.core.display import clear_output
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import copy

from prioritized__experience_replay import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):

    def __init__(self, input_shape, action_space):
        super(Model, self).__init__()
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


class Agent:

    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.env.seed(42)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        self.per_memory = PrioritizedExperienceReplay(2000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.0005

        self.batch_size = 32
        self.train_start = 1000
        self.target_sync = 20

        self.model = Model(input_shape=self.state_size, action_space=self.action_size).float()

        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.ddqn = True
        self.epsilon_greedy = True
        self.PER_use = True

    def remember(self, state, action, reward, next_state, done):
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        if self.PER_use:
            self.per_memory.push((state, next_state, action, reward, done))
        else:
            self.memory.append((state, next_state, action, reward, done))

    def act(self, state, decay_step):
        if self.epsilon_greedy:
            explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
                -self.epsilon_decay * decay_step)
        else:
            if self.epsilon > self.epsilon_min and len(self.memory) > self.train_start:
                self.epsilon *= self.epsilon_decay
            explore_probability = self.epsilon

        if explore_probability > np.random.rand():
            # Make a random action (exploration)
            return random.randrange(self.action_size)
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            return torch.argmax(self.model(state, model='online')).item()

    def replay(self):
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

        online = self.model(state, model='online')
        ab = np.arange(0, self.batch_size)
        online = online[
            ab, action
        ]

        if self.ddqn:
            target_next = self.model(next_state, model='target')
            best_action = torch.argmax(target_next, axis=1)

            next_Q = self.model(next_state, model='target')[
                np.arange(0, self.batch_size), best_action
            ]
            td = (reward + (1 - done.float()) * self.gamma * next_Q).float()

        else:
            target_next = self.model(state, model='online')
            best_action = torch.argmax(target_next, axis=1)

            next_Q = self.model(state, model='online')[
                np.arange(0, self.batch_size), best_action
            ]
            td = (reward + (1 - done.float()) * self.gamma * next_Q).float()

        loss = self.loss_fn(online, td)

        if self.PER_use:
            # indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = (online - next_Q).abs()
            # Update priority
            self.per_memory.update_priorities(tree_idx, absolute_errors)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        if self.ddqn:
            self.model.target.load_state_dict(self.model.online.state_dict())

    def moving_average(self, x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    def plot_res(self, values, title=''):
        ''' Plot the reward curve and histogram of results over time.'''
        # Update the window after each episode
        clear_output(wait=True)

        # Define the figure
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        f.suptitle(title)
        ax[0].plot(values, label='score per run')
        ax[0].axhline(195, c='red', ls='--', label='goal')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Reward')
        x = range(len(values))
        # Calculate the trend

        if len(x) > 15:
            ax[0].plot(x[14:], self.moving_average(values, 15), label='15 moving average')
        try:
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            ax[0].plot(x, p(x), "--", label='trend')

        except:
            print('')
        ax[0].legend()

        # Plot the histogram of results
        ax[1].hist(values[-50:])
        ax[1].axvline(195, c='red', label='goal')
        ax[1].set_xlabel('Scores per Last 50 Episodes')
        ax[1].set_ylabel('Frequency')
        ax[1].legend()
        plt.show()

    def run(self):
        final = []
        decay_step = 0
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float)
            done = False
            i = 0
            total = 0
            while not done:
                self.env.render()
                decay_step += 1
                action = self.act(state, decay_step)
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float)
                total += reward
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    # print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                self.replay()
            if e % self.target_sync == 0:
                self.update_target()
            # if e % 10 == 0:
                # final.append(total)
                # self.plot_res(final)


if __name__ == "__main__":
    agent = Agent()
    agent.run()
