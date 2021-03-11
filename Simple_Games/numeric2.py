import gym
import math
import random
import numpy as np
import time
import pickle
import keyboard

from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

from copy import copy, deepcopy
from itertools import count
from collections import deque

import torchvision.transforms as T
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

resize = T.Compose([T.ToPILImage(),
                    T.CenterCrop((250, 500)),
                    T.Resize(64),
                    T.Grayscale(),
                    T.ToTensor()])

# env = gym.make('CartPole-v0').unwrapped
env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
hidden_dim = 64

model_name = 'Dueling_DDQN_Prior_Memory'
save_name = 'checkpoints/' + model_name
resume = False


class Config():

    def __init__(self):
        self.n_episode = 1000
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 10
        self.TARGET_UPDATE = 200  # over steps
        self.BATCH_SIZE = 256
        self.start_from = 512
        self.GAMMA = 0.95 #1
        # self.dueling = True
        self.plot_every = 50
        self.lr = 2e-4
        self.optim_method = optim.Adam
        self.memory_size = 10000



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple(
    'Transition', ['state', 'action', 'reward', 'next_state', 'terminal'])


class ReplayMemory(object):
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.prob_alpha = alpha
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0 ** self.prob_alpha

        total = len(self.buffer)
        if total < self.capacity:
            pos = total
            self.buffer.append(transition)
        else:
            prios = self.priorities[:total]
            probs = (1 - prios / prios.sum()) / (total - 1)
            pos = np.random.choice(total, 1, p=probs)

        self.priorities[pos] = max_prio

    def sample(self, batch_size):
        total = len(self.buffer)
        prios = self.priorities[:total]
        probs = prios / prios.sum()

        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min * total) ** (-beta)

        weights = (total * probs[indices]) ** (-beta)
        weights /= max_weight
        weights = torch.tensor(weights, device=device, dtype=torch.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + 1e-5) ** self.prob_alpha

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(input_dim, hidden_dim)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(hidden_dim, output_dim)

        # Define sigmoid activation
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.output(x)

        return x

class History():

    def __init__(self, plot_size=300, plot_every=5):
        self.plot_size = plot_size
        self.episode_durations = deque([], self.plot_size)
        self.means = deque([], self.plot_size)
        self.episode_loss = deque([], self.plot_size)
        self.indexes = deque([], self.plot_size)
        self.step_loss = []
        self.step_eps = []
        self.peak_reward = 0
        self.peak_mean = 0
        self.moving_avg = 0
        self.step_count = 0
        self.total_episode = 0
        self.plot_every = plot_every

    def update(self, t, episode_loss):
        self.episode_durations.append(t + 1)
        self.episode_loss.append(episode_loss / (t + 1))
        self.indexes.append(self.total_episode)
        if t + 1 > self.peak_reward:
            self.peak_reward = t + 1
        if len(self.episode_durations) >= 100:
            self.means.append(sum(list(self.episode_durations)[-100:]) / 100)
        else:
            self.moving_avg = self.moving_avg + (t - self.moving_avg) / (self.total_episode + 1)
            self.means.append(self.moving_avg)
        if self.means[-1] > self.peak_mean:
            self.peak_mean = self.means[-1]

        if self.total_episode % self.plot_every == 0:
            self.plot()

    def plot(self):
        # display.clear_output(wait=True)

        f, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.plot(self.indexes, self.episode_durations)
        ax1.plot(self.indexes, self.means)
        ax1.axhline(self.peak_reward, color='g')
        ax1.axhline(self.peak_mean, color='g')

        ax2 = ax1.twinx()
        ax2.plot(self.indexes, self.episode_loss, 'r')

        ax4 = ax3.twinx()
        total_step = len(self.step_loss)
        sample_rate = total_step // self.plot_size if total_step > (
                self.plot_size * 10) else 1
        ax3.set_title('total: {0}'.format(total_step))
        ax3.plot(self.step_eps[::sample_rate], 'g')
        ax4.plot(self.step_loss[::sample_rate], 'b')

        plt.pause(0.00001)


def optimize_model(step):
    if len(memory) < config.start_from:
        return 0

    if step % config.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Sample memory as a batch
    samples, ids, weights = memory.sample(config.BATCH_SIZE)
    batch = Transition(*zip(*samples))

    # A tensor cannot be None, so strip out terminal states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None], dim=0)

    state_batch = torch.stack(batch.state, dim=0)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Bellman's Equation
    with torch.no_grad():
        # weights: size = (out_features, in_features)
        # input matmul weight
        # result : size = (batch_size, ..., out_features)
        online_Q = policy_net(non_final_next_states)  # arg: size: (batch_size, ...,  in_features)
        target_Q = target_net(non_final_next_states)
        next_Q = torch.zeros(config.BATCH_SIZE, device=device)
        next_Q[non_final_mask] = target_Q.gather(
            1, online_Q.max(1)[1].detach().unsqueeze(1)).squeeze(1)
        target_Q = next_Q * config.GAMMA + reward_batch

    # Compute loss
    policy_net.train()
    current_Q = policy_net(state_batch).gather(1, action_batch)
    diff = current_Q.squeeze() - target_Q
    loss = (0.5 * (diff * diff) * weights).mean()

    # Update memory
    delta = diff.abs().detach().cpu().numpy().tolist()
    memory.update_priorities(ids, delta)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy()


def epsilon_by_frame(frame_idx):
    return config.epsilon_final + \
           (config.epsilon_start - config.epsilon_final) * math.exp(-1. * frame_idx / config.epsilon_decay)


def select_action(state, eps):
    sample = random.random()
    if sample > eps:
        policy_net.eval()
        with torch.no_grad():
            # print(policy_net(state))
            # print(policy_net(state).max(0))  # arg of max sets which dim must be reduced
            # print(policy_net(state).max(0)[1].view(1, 1))  # [1] for selecting 2nd arg returned by max: indexes
            # print()
            return policy_net(state).max(0)[1].view(1, 1)
        # return A
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


# comment this out if a checkpoint is available
# load_name = 'checkpoints/checkpoint'

def training():

    global memory
    global config
    global target_net
    global policy_net
    global optimizer


    # Init network
    if resume:
        print('loading checkpoint...')
        with open(save_name + '.pickle', 'rb') as f:
            data = pickle.load(f)
            history = data['history']
            config = data['config']

        checkpoint = torch.load(save_name + '.pt')

        policy_net = DQN().to(device)
        target_net = DQN().to(device)
        policy_net.load_state_dict(checkpoint['policy_net'])
        target_net.load_state_dict(checkpoint['target_net'])

        optimizer = config.optim_method(policy_net.parameters(), lr=config.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('fresh start...')
        history = History()
        config = Config()

        policy_net = DQN().to(device)
        print(policy_net)
        target_net = deepcopy(policy_net)
        optimizer = config.optim_method(policy_net.parameters(), lr=config.lr)

    memory = ReplayMemory(config.memory_size)
    target_net.eval()

    for i_episode in range(config.n_episode):
        history.total_episode += 1

        obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0

        state = torch.tensor(obs, device=device, dtype=torch.float32)
        avg_loss = 0

        if (i_episode > 0):
            print('episode: ', i_episode, " ", history.means[-1])
            av_lenghs.append(history.means[-1])

            if history.means[-1] == 199.0:
                break

        for t in count():

            if keyboard.is_pressed('p'):
                print("Paused")
                keyboard.wait('p')
                print("Resumed")
                time.sleep(1)

            if keyboard.is_pressed('s'):
                print("Stopped")
                print()
                return

            history.step_count += 1

            # Select and perform an action
            eps = epsilon_by_frame(history.total_episode)
            action = select_action(state, eps)
            history.step_eps.append(eps)
            obs_next, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Do optimization in main thread
            loss = optimize_model(history.step_count)
            avg_loss += loss
            history.step_loss.append(loss)

            # Render the next_state and remember it
            # append screens
            next_state = torch.tensor(obs_next, device=device, dtype=torch.float32) if not done else None
            memory.push(Transition(state, action, reward, next_state, done))

            # Move to the next state
            state = next_state

            if done:
                history.update(t, avg_loss)
                break


av_lenghs = []

for run in range(1):
    print()
    print("run nb: ", run)
    av_lenghs.append(-1)
    training()
    torch.cuda.empty_cache()


n = 1
r = 0
file = open("durations", "w+")
for i in av_lenghs:
    if i == -1:
        file.write("run nb: " + str(r) + '\n')
        r += 1
        n = 1
    else:
        string = str(n) + " " + str(i) + '\n'
        file.write(string)
        n += 1

file.close()


def testing():
    env = gym.make('CartPole-v0').unwrapped
    for i in range(100):
        obs, done, rew = env.reset(), False, 0
        state = torch.tensor(obs, device=device, dtype=torch.float32)

        total_reward = 0

        while True:
            # Select and perform an action
            action = select_action(state, 0)
            obs, reward, done, info = env.step(action.item())
            state = torch.tensor(obs, device=device, dtype=torch.float32)
            total_reward += reward
            time.sleep(0.001)
            env.render()

            if done:
                print('total reward:', total_reward)
                break


testing()
