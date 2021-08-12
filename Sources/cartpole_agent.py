import random
import gym as gym
import datetime

from preprocessing import *
from agent import *
from Logger import *


class CartpoleAgent(Agent):
    # Concrete class extending the functionality of Agent

    def __init__(self, network, config):
        self.env = gym.make('CartPole-v1')

        self.state_size = self.env.observation_space
        self.number_actions = self.env.action_space.n
        super(CartpoleAgent, self).__init__(network, config, self.env.observation_space.shape[0], self.number_actions)

        save_dir = Path("checkpoints_cartpole") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        self.logger = MetricLogger(save_dir)

    def reset(self):
        state = self.env.reset()
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
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)  # ?device?

        return next_state, reward, done, info, action
