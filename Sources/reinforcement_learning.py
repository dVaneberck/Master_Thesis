import gym as gym
import argparse
import sys
import yaml

from minecraft_agent import *
from mario_agent import *
from cartpole_agent import *


def train(agent):

    decay_step = 0
    for ep in range(agent.nEpisodes):

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

            if agent.more_info:
                print("total reward: ", total, ", reward: ", reward)
                if info:
                    print("info: ", info)
                print()
            if done:
                print("episode: {}/{}, life time: {}".format(ep, agent.nEpisodes, alive_step))
                agent.write_rewards(ep, total)

        if ep % agent.target_sync == 0:
            agent.update_target()
        agent.logger.log_episode()
        agent.logger.record(episode=ep, epsilon=agent.explore_probability, step=decay_step)


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
