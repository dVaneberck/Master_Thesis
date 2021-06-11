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
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import datetime

from prioritized__experience_replay import *

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

import argparse
from pathlib import Path
import time
from PIL import Image

from Logger import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ChooseObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space["compassAngle"]

    def observation(self, observation):
        print(observation["compassAngle"])  #debug
        return observation["compassAngle"]


class Model(nn.Module):

    def __init__(self, input_shape, action_space):
        super(Model, self).__init__()
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


class Agent:

    def __init__(self):
        self.env = gym.make('MineRLNavigateDense-v0')
        self.use_cuda = torch.cuda.is_available()

        self.nFrames = 2
        self.env = SkipFrame(self.env, skip=3)
        self.env = ChooseObservation(self.env)
        self.env = FrameStack(self.env, num_stack=self.nFrames)

        self.state_size = self.env.observation_space
        self.action_size = self.env.action_space
        self.enum_actions = {0: "camera", 1: "camera2", 2: "jumpfront", 3: "forward", 4: "jump", 5: "back",
                             6: "left", 7: "right", 8: "sprint", 9: "sneak", 10: "place", 11: "attack"}
        # self.number_actions = len(self.enum_actions)
        self.number_actions = 3

        self.EPISODES = 10000000
        self.memory = deque(maxlen=2000)
        self.per_memory = PrioritizedExperienceReplay(10000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.0005
        self.current_step = 0

        self.batch_size = 32
        self.train_start = 1000
        self.target_sync = 20

        self.model = Model(input_shape=self.nFrames, action_space=self.number_actions).float()  # 4 because 4 states in time
        self.model = self.model.to(device)

        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.loss_fn = torch.nn.SmoothL1Loss().to(device=device)

        self.ddqn = True
        self.epsilon_greedy = True
        self.PER_use = True

        save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        self.logger = MetricLogger(save_dir)

    def remember(self, state, action, reward, next_state, done):
        if device.type == 'cuda':
            state = torch.tensor(state, device=device)
            next_state = torch.tensor(next_state, device=device)
            action = torch.tensor([action], device=device)
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)
        else:
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
            self.current_step += 1
            return random.randrange(self.number_actions)
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)

            state = state.unsqueeze(0)
            self.current_step += 1
            q_values = self.model(state, model='online')
            best_q, best_action = torch.max(q_values, dim=1)

            return best_action.item()

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

    def run(self):
        use_cuda = torch.cuda.is_available()
        print(f"Using CUDA: {use_cuda}")

        open("rewards.txt", "w").close()
        open("compass.txt", "w").close()

        decay_step = 0
        for e in range(self.EPISODES):
            compass_array = []

            state = self.env.reset()
            state = state.__array__()
            state = torch.tensor(state, dtype=torch.float, device=device)  # ?device?

            done = False
            i = 0
            total = 0
            while not done:
                self.env.render()
                decay_step += 1
                compass = state.data[0].item()
                compass_array.append(compass)

                # enum_actions = {0: "camera", 1: "camera2", 2: "jumpfront", 3: "forward", 4: "jump", 5: "back",
                #                 6: "left", 7: "right", 8: "sprint", 9: "sneak", 10: "place", 11: "attack"}

                action_basic = self.act(state, decay_step)
                action = self.env.action_space.noop()
                if action_basic == 2:
                    action['jump'] = 1
                    action['forward'] = 1
                    print("jump forward")
                elif action_basic == 0:
                    action['camera'] = [0, 1]  # turn camera 1 degrees right for this step
                    print("turn right")
                elif action_basic == 1:
                    action['camera'] = [0, -1]
                    print("turn left")
                elif action_basic == 8:
                    action['sprint'] = 1
                    action['forward'] = 1
                elif action_basic == 10:
                    action['place'] = 'dirt'
                else:
                    action[self.enum_actions[action_basic]] = 1

                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.__array__()
                next_state = torch.tensor(next_state, dtype=torch.float, device=device) # ?device?
                total += reward

                print("total reward: ", total, ", reward: ", reward)
                if info:
                    print("info: ", info)
                print()

                self.remember(state, action_basic, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    # print("episode: {}/{}, life time: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    print("episode: {}/{}, life time: {}".format(e, self.EPISODES, i))

                    save_file = open("rewards.txt", "a")
                    save_file.write(str(e) + " " + str(total) + '\n')
                    save_file.close()

                    compass_file = open("compass.txt", "a")
                    compass_file.write(str(compass_array) + '\n')
                    compass_file.close()

                self.replay(reward)

            if e % self.target_sync == 0:
                self.update_target()
            self.logger.log_episode()
            if e % 100 == 0:
                self.logger.record(episode=e, epsilon=self.epsilon, step=self.current_step)


if __name__ == "__main__":
    device = torch.device("cuda")
    agent = Agent()
    agent.run()
