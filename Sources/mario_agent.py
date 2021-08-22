import random
import gym as gym
from gym.wrappers import FrameStack
import datetime

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
        super(MarioAgent, self).__init__(network, config, 3136, self.number_actions)
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
            self.explore_probability = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
                -self.epsilon_decay * decay_step)
        else:
            if self.epsilon_max > self.epsilon_min and len(self.memory) > self.train_start:
                self.epsilon_max *= self.epsilon_decay
            self.explore_probability = self.epsilon_max

        if self.explore_probability > np.random.rand():
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
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)

        return next_state, reward, done, info, action
