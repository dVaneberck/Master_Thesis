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

from prioritized__experience_replay import *
from neural_net import *
from preprocessing import *
from agent import *
from minecraft_agent import *
from Logger import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(agent):

    open("rewards.txt", "w").close()

    decay_step = 0
    for e in range(agent.EPISODES):

        state = agent.env.reset()
        state = state.__array__()
        state = torch.tensor(state, dtype=torch.float, device=device)  # ?device?

        done = False
        i = 0
        total = 0
        while not done:
            agent.env.render()
            decay_step += 1

            next_state, reward, done, info, action = agent.act(state, decay_step)

            next_state = next_state.__array__()
            next_state = torch.tensor(next_state, dtype=torch.float, device=device)  # ?device?
            total += reward

            print("total reward: ", total, ", reward: ", reward)
            if info:
                print("info: ", info)
            print()

            agent.store(state, action, reward, next_state, done)
            state = next_state
            i += 1
            if done:
                print("episode: {}/{}, life time: {}".format(e, agent.EPISODES, i))

                save_file = open("rewards.txt", "a")
                save_file.write(str(e) + " " + str(total) + '\n')
                save_file.close()

            agent.update_net(reward)

        if e % agent.target_sync == 0:
            agent.update_target()
        agent.logger.log_episode()
        agent.logger.record(episode=e, epsilon=agent.epsilon, step=agent.current_step)


def main():
    # Decode program argument:
    parser = argparse.ArgumentParser(description='Run a reinforcement learning algorithm')

    parser.add_argument("game_type", help=
    "The type of game that should be learned. Can be either 'cartpole', 'mario' or 'minecraft'")

    parser.add_argument("-network", type=int, help="Type of input. Either 1 for numerical, or 2 for convolutional")

    args = parser.parse_args()

    net = None
    choose_obs = None
    if args.network == 1:
        net = SmallMLP
        choose_obs = ChooseCompassObservation
    elif args.network == 2:
        net = ConvNet
        choose_obs = ChoosePovObservation
    else:
        print("The type of network entered is not recognized")
        exit(0)

    agent = None
    if args.game_type == "minecraft":
        agent = MinecraftAgent(net, choose_obs)
    elif args.game_type == "mario":
        pass
    elif args.game_type == "cartpole":
        pass
    else:
        print("The type of game entered is not recognized")
        exit(0)

    print(f"Using CUDA: {agent.use_cuda}")
    print("Using environment: " + str(agent.env.unwrapped))
    print("\nTraining: ")

    # run main training loop:
    train(agent)


if __name__ == "__main__":
    main()
