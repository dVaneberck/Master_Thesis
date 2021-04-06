import minerl
import gym


def main():
    # do your main minerl code
    env = gym.make('MineRLNavigateDense-v0')

    obs = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()

        # One can also take a no_op action with
        # action =env.action_space.noop()

        obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    main()