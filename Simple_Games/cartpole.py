import math
import random

import gym as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import prioritized__experience_replay as PER
import cv2

env = gym.make('CartPole-v0').unwrapped
env.seed(42)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tuple representing a single transition:
Transition = namedtuple(
    'Transition2', ['state', 'action', 'reward', 'next_state', 'terminal'])


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - kernel_size + 4) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))))

        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return self.head(x.view(x.size(0), -1))


resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])


def get_screen(show=False):
    screen = env.render(mode='rgb_array')
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    screen = cv2.resize(screen, (240, 160), interpolation=cv2.INTER_CUBIC)
    screen[screen < 255] = 0
    screen = screen / 255
    screen = torch.from_numpy(screen)
    screen = resize(screen).unsqueeze(0).to(device)
    if show:
        plt.figure()
        plt.imshow(screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
                   interpolation='none', cmap='gray')
        plt.title('Extracted screen')
        plt.show()
    return screen


env.reset()

BATCH_SIZE = 256
GAMMA = 1.0
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10
TARGET_UPDATE = 200

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = None
target_net = None

optimizer = None
memory = None

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    # steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf(), block=False)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions, ids, _ = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    diff = state_action_values.squeeze() - expected_state_action_values
    delta = diff.abs().detach().cpu().numpy().tolist()
    memory.update_priorities(ids, delta)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def exec_cartpole():
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape

    global policy_net, target_net
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    global optimizer, memory
    # optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), 3e-5)

    memory = PER.PrioritizedExperienceReplay(10000)

    num_episodes = 50000
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        screens = deque([init_screen] * 4, 4)
        state = torch.cat(list(screens), dim=1)

        global steps_done
        steps_done += 1
        for t in count():
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            screens.append(get_screen())
            if not done:
                next_state = torch.cat(list(screens), dim=1)
            else:
                next_state = None

            memory.push(Transition(state, action, reward, next_state, done))
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()
