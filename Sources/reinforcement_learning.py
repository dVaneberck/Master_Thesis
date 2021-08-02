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
import torchvision.transforms as T
import datetime
import argparse
from pathlib import Path
import time
import sys
import yaml

from prioritized__experience_replay import *
from neural_net import *
from preprocessing import *
from agent import *
from minecraft_agent import *
from mario_agent import *
from cartpole_agent import *
from Logger import *


def train(agent):

    open("rewards.txt", "w").close()

    decay_step = 0
    for ep in range(agent.EPISODES):

        state = agent.reset()

        done = False
        alive_step = 0
        total = 0
        while not done:
            decay_step += 1
            alive_step += 1

            next_state, reward, done, info, action = agent.act(state, decay_step)
            total += reward
            agent.store(state, action, reward, next_state, done)
            agent.update_net(reward)

            state = next_state

            print("total reward: ", total, ", reward: ", reward)
            if info:
                print("info: ", info)
            print()
            if done:
                print("episode: {}/{}, life time: {}".format(ep, agent.EPISODES, alive_step))

                save_file = open("rewards.txt", "a")
                save_file.write(str(ep) + " " + str(total) + '\n')
                save_file.close()

        if ep % agent.target_sync == 0:
            agent.update_target()
        agent.logger.log_episode()
        agent.logger.record(episode=ep, epsilon=agent.epsilon, step=decay_step)


def main():
    # Decode program argument, and launch adequate agent accordingly :
    parser = argparse.ArgumentParser(description='Run a reinforcement learning algorithm')

    parser.add_argument("config_file", help="path of the configuration file that specify all parameters")

    args = parser.parse_args()

    with open(args.config_file) as file:
        config = yaml.safe_load(file)

    if config["network"] == 'SmallMLP':
        net = SmallMLP
        choose_obs = ChooseCompassObservation
    elif config["network"] == 'BigMLP':
        net = BigMLP
        choose_obs = ChooseCompassObservation
    elif config["network"] == 'ConvNet':
        net = ConvNet
        choose_obs = ChoosePovObservation
    else:
        print('The type of network entered is not recognized')
        sys.exit()

    if config["game"] == "minecraft":
        agent = MinecraftAgent(net, config, choose_obs)
    elif config["game"] == "mario":
        agent = MarioAgent(ConvNet, config)
    elif config["game"] == "cartpole":
        agent = CartpoleAgent(net, config)
    else:
        print('The type of game entered is not recognized')
        sys.exit()

    print(f"Using CUDA: {agent.use_cuda}")
    print("Using environment: " + str(agent.env.unwrapped))
    print("Using neural network: " + type(agent.model).__name__)
    print("\nTraining: ")

    # run main training loop:
    train(agent)


if __name__ == "__main__":
    main()
