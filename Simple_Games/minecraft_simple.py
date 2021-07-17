import minerl
import math
import random
import time

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


from Logger import *

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


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        # self.observation_space.shape = obs_shape

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        print("------------------------")
        print("compass angle: ", observation["compassAngle"])
        observation = self.permute_orientation(observation["pov"])
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation



class Agent:

    def __init__(self):
        # self.env.seed(42)
        self.env = gym.make('MineRLNavigateDense-v0')

        self.env.observation_space = self.env.observation_space["pov"]
        # self.env.reset()
        self.env = SkipFrame(self.env, skip=3)
        # self.env = GrayScaleObservation(self.env)
        # self.env = ResizeObservation(self.env, shape=self.env.observation_space.shape[0])
        # self.env = FrameStack(self.env, num_stack=4)

        self.state_size = self.env.observation_space
        self.action_size = self.env.action_space
        self.enum_actions = {0: "back", 1: "forward",  2: "left", 3: "right", 4: "jump", 5: "sprint",
                        6: "camera", 7: "camera2", 8: "jumpfront", 9: "sneak", 10: "place", 11: "attack"}
        self.number_actions = len(self.enum_actions)-3

        self.EPISODES = 10000000



    def run(self):

        decay_step = 0
        reward = 0
        for e in range(self.EPISODES):

            self.env.seed(3)
            state = self.env.reset()


            done = False
            i = 0
            DEBUG = 0
            total = 0
            while not done:
                self.env.render()
                decay_step += 1
                time.sleep(0.1)

                compass = state["compassAngle"]
                pov = state["pov"]

                print("----------------")
                print(compass)

                # enum_actions = {0: "back", 1: "forward", 2: "left", 3: "right", 4: "jump", 5: "sprint",
                #                 6: "camera", 7: "camera2", 8: "jumpfront", 9: "sneak", 10: "place", 11: "attack"}
                action_basic = 0

                action = self.env.action_space.noop()


                if -5 < compass < 5:
                    action['forward'] = 1
                elif compass < -5:
                    action['camera'] = [0, -1]
                    action['forward'] = 1
                    print("left")
                elif compass > 5:
                    action['camera'] = [0, +1]
                    action['forward'] = 1
                    print("right")

                if reward == 0:
                    action['jump'] = 1
                action['jump'] = 1

                if i == 0:
                    action['camera'] = [9, 0]

                if DEBUG != 0:
                    action = self.env.action_space.noop()
                    if DEBUG == 1:
                        action['forward'] = 1
                        action['jump'] = 1
                    elif DEBUG == 2:
                        action['left'] = 1
                        action['jump'] = 1
                    elif DEBUG == 3:
                        action['right'] = 1
                        action['jump'] = 1
                    elif DEBUG == 4:
                        action['back'] = 1
                        action['jump'] = 1
                    elif DEBUG == 5:
                        action = self.env.action_space.noop()

                next_state, reward, done, info = self.env.step(action)

                state = next_state

                total += reward

                print("total reward: ", total, ", reward: ", reward)
                if info:
                    print("info: ", info)

                i += 1
                if done:
                    # print("episode: {}/{}, life time: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    print("info: ", info)
                    print("episode: {}/{}, life time: {}, total rew: {}".format(e, self.EPISODES, i, total))
                    if total > 100:
                        print("goal reached")



if __name__ == "__main__":
    device = torch.device("cuda")
    agent = Agent()
    agent.run()
