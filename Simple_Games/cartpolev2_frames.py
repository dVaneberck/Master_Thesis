import math
import random

import gym as gym
from gym.wrappers import FrameStack
from gym.spaces import Box
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from IPython.core.display import clear_output
from PIL import Image
import torchvision.transforms as T

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import copy

from prioritized__experience_replay import *

from pathlib import Path
from Logger import *

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):

    def __init__(self, input_shape, action_space):
        super(Model, self).__init__()
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10240, 512),
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
        self.EPISODES = 1000000
        self.memory = deque(maxlen=2000)
        self.per_memory = PrioritizedExperienceReplay(2000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.0005
        self.current_step = 0

        self.batch_size = 32
        self.train_start = 1000
        self.target_sync = 20

        self.model = Model(input_shape=self.state_size, action_space=self.action_size).float()

        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.loss_fn = torch.nn.MSELoss()

        self.ddqn = True
        self.epsilon_greedy = True
        self.PER_use = True

        save_dir = Path("checkpoints_cartpole") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        self.logger = MetricLogger(save_dir)

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

        self.current_step += 1
        if explore_probability > np.random.rand():
            # Make a random action (exploration)
            return random.randrange(self.action_size)
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            state = state.unsqueeze(0)
            return torch.argmax(self.model(state, model='online'), axis=1).item()

    def replay(self, reward_log):
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
            target_next = self.model(next_state, model='online')
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

        test = reward_log
        self.logger.log_step(reward=test, loss=loss.item(), q=online.mean().item())

    def update_target(self):
        if self.ddqn:
            self.model.target.load_state_dict(self.model.online.state_dict())

    # def to_gray(self, state):
    #     gray = [0.2989, 0.5870, 0.1140]
    #     state = np.dot(state[..., :3], gray)
    #     return state

    # def screen_resize(self, state):
    #     _, height, width = state.shape
    #     state = state[:, int(height*0.6):int(width*0.8)]
    #     plt.imshow(state.transpose(1, 2, 0), interpolation='nearest')
    #     plt.show()
    #     return state

    def process_screen(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, dsize=(120, 80), interpolation=cv2.INTER_CUBIC)
        state[state < 255] = 0
        return state

    def run(self):
        decay_step = 0
        t0 = time.perf_counter()
        last_rewards = deque(maxlen=10)
        # self.env.reset()

        for e in range(self.EPISODES):
            if (time.perf_counter() - t0) / 60 > 20:
                break
            self.env.reset()

            state = self.env.render(mode='rgb_array')
            state = self.process_screen(state)

            # plt.imshow(state, interpolation='nearest')
            # plt.show()

            state = state[..., np.newaxis]
            state = np.append(state, state, 2)
            state = np.append(state, state, 2)
            state = state.transpose((2, 0, 1))

            state_tensor = np.ascontiguousarray(state)
            state_tensor = torch.from_numpy(state_tensor)
            state_tensor = torch.tensor(state_tensor, dtype=torch.float)

            done = False
            i = 0
            total = 0
            while not done:
                self.env.render()
                decay_step += 1
                action = self.act(state_tensor, decay_step)
                _, reward, done, _ = self.env.step(action)

                next_state = self.env.render(mode='rgb_array')
                next_state = self.process_screen(next_state)
                next_state = next_state[..., np.newaxis]
                next_state = next_state.transpose((2, 0, 1))
                next_state = np.append(state, next_state, 0)
                next_state = np.delete(next_state, 0, 0)
                state = next_state

                next_state = np.ascontiguousarray(next_state)
                next_state = torch.from_numpy(next_state)
                next_state = torch.tensor(next_state, dtype=torch.float)

                total += reward
                self.remember(state_tensor, action, reward, next_state, done)

                state_tensor = np.ascontiguousarray(state)
                state_tensor = torch.from_numpy(state_tensor)
                state_tensor = torch.tensor(state_tensor, dtype=torch.float)

                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                self.replay(reward)
            last_rewards.append(total)
            self.logger.log_episode()
            if e % self.target_sync == 0:
                self.update_target()
            if e % 10 == 0:
                self.logger.log_times((time.perf_counter() - t0) / 60)
                self.logger.record(episode=e, epsilon=self.epsilon, step=self.current_step, times=True)


if __name__ == "__main__":
    agent = Agent()
    agent.run()
