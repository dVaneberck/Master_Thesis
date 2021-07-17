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
from PIL import Image

from prioritized__experience_replay import *
from neural_net import *
from preprocessing import *
from Logger import *

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace
# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, game_type):
        self.game_type = game_type
        if game_type == "minecraft":
            self.env = gym.make('MineRLNavigateDense-v0')
        elif game_type == "mario":
            self.env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        elif game_type == "cartpole":
            self.env = gym.make('CartPole-v1')

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

        self.EPISODES = 400
        self.memory = deque(maxlen=2000)
        self.per_memory = PrioritizedExperienceReplay(10000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.0005
        self.current_step = 0

        self.batch_size = 32
        self.train_start = 1000
        self.target_sync = 2  # in episodes

        self.model = NeuralNet(input_shape=self.nFrames, action_space=self.number_actions).float()
        self.model = self.model.to(device)

        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.loss_fn = torch.nn.SmoothL1Loss().to(device=device)

        self.ddqn = True
        self.epsilon_greedy = True
        self.PER_use = True

        save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        self.logger = MetricLogger(save_dir)

    def store(self, state, action, reward, next_state, done):
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
            # indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = (online_Q - next_Q).abs()
            # Update priority
            self.per_memory.update_priorities(tree_idx, absolute_errors)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        test = reward_log
        self.logger.log_step(reward=test, loss=loss.item(), q=online_Q.mean().item())

    def update_target(self):
        if self.ddqn:
            self.model.target.load_state_dict(self.model.online.state_dict())

    def train(self):
        print(f"Using CUDA: {self.use_cuda}")

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

                self.store(state, action_basic, reward, next_state, done)
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

                self.update_net(reward)

            if e % self.target_sync == 0:
                self.update_target()
            self.logger.log_episode()
            # if e % 100 == 0:
            self.logger.record(episode=e, epsilon=self.epsilon, step=self.current_step)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run a reinforcement learning algorithm')

    parser.add_argument("game_type", help=
                        "the type of game that should be learned. Can be either 'cartpole', 'mario' or 'minecraft'")
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    args = parser.parse_args()
    if args.game_type not in ("cartpole", "mario", "minecraft"):
        print("The type of game entered is not recognized")
        exit(0)

    agent = Agent(args.game_type)

    print(f"Using CUDA: {agent.use_cuda}")
    print("Using environment: " + str(agent.env.unwrapped))
    print("\nTraining: ")
    agent.train()
