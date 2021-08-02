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
from PIL import Image
import torchvision.transforms as T
import datetime
import argparse
from pathlib import Path
import time
from PIL import Image

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from preprocessing import *
from agent import *
from Logger import *


class MarioAgent(Agent):
    # Concrete class extending the functionality of Agent

    def __init__(self, network, config):
        self.env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

        self.env = JoypadSpace(self.env, [["right"], ["right", "A"]])
        self.env = SkipFrame(self.env, skip=config["skipFrames"])
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=84)
        self.number_actions = self.env.action_space.n
        super(MarioAgent, self).__init__(network, 3136, self.number_actions)
        self.env = FrameStack(self.env, num_stack=self.nFrames)
        self.state_size = self.env.observation_space

        save_dir = Path("checkpoints_mario") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        self.logger = MetricLogger(save_dir)

    def reset(self):
        state = self.env.reset()
        state = state.__array__()
        return torch.tensor(state, dtype=torch.float, device=self.device)

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
            action = random.randrange(self.number_actions)
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)

            state = state.unsqueeze(0)
            q_values = self.model(state, model='online')
            best_q, best_action = torch.max(q_values, dim=1)

            action = best_action.item()

        if self.render:
            self.env.render()

        next_state, reward, done, info = self.env.step(action)
        next_state = next_state.__array__()
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)  # ?device?

        return next_state, reward, done, info, action

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

        online = self.model(state, model='online')   # all Q-values predicted
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
