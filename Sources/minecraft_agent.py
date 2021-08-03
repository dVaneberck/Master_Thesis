import minerl
import gym as gym
from gym.wrappers import FrameStack

from preprocessing import *
from agent import *
from Logger import *


class MinecraftAgent(Agent):
    # Concrete class extending the functionality of Agent

    def __init__(self, network, config, choose_obs):
        self.enum_actions = {0: "camera", 1: "camera2", 2: "jumpfront", 3: "forward", 4: "jump", 5: "back",
                             6: "left", 7: "right", 8: "sprint", 9: "sneak", 10: "place", 11: "attack"}

        self.number_actions = config["number_actions"]
        self.compass_array = []

        super(MinecraftAgent, self).__init__(network, config, 1024, self.number_actions)
        self.env = gym.make('MineRLNavigateDense-v0')

        self.env = SkipFrame(self.env, skip=config["skipFrames"])
        self.env = choose_obs(self.env, self)
        if isinstance(self.model, ConvNet):
            self.env = GrayScaleObservation(self.env)
            self.env = ResizeObservation(self.env, shape=self.env.observation_space.shape[0])
        self.env = FrameStack(self.env, num_stack=self.nFrames)

        self.state_size = self.env.observation_space
        self.action_size = self.env.action_space

        save_dir = Path("checkpoints_minerl") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        self.logger = MetricLogger(save_dir)

        # erase and create new compass file:
        self.compass_file = config["compass_file"]
        open(self.compass_file, "w").close()

    def reset(self):
        if self.fix_seed:
            self.env.seed(42)
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
            action_basis = random.randrange(self.number_actions)
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)

            state = state.unsqueeze(0)
            q_values = self.model(state, model='online')
            best_q, best_action = torch.max(q_values, dim=1)

            action_basis = best_action.item()

        if self.more_info:
            print(self.compass_array[-1])

        # format the action for minerl:
        action = self.env.action_space.noop()
        if action_basis == 2:
            action['jump'] = 1
            action['forward'] = 1
            print("jump forward")
        elif action_basis == 0:
            action['camera'] = [0, 1]  # turn camera 1 degrees right for this step
            print("turn right")
        elif action_basis == 1:
            action['camera'] = [0, -1]
            print("turn left")
        elif action_basis == 8:
            action['sprint'] = 1
            action['forward'] = 1
        elif action_basis == 10:
            action['place'] = 'dirt'
        else:
            action[self.enum_actions[action_basis]] = 1

        if self.render:
            self.env.render()

        next_state, reward, done, info = self.env.step(action)
        next_state = next_state.__array__()
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)  # ?device?

        if done:
            self.write_compass()

        return next_state, reward, done, info, action_basis

    def write_compass(self):
        compass_file = open(self.compass_file, "a")
        compass_file.write(str(self.compass_array) + '\n')
        compass_file.close()
        self.compass_array = []
