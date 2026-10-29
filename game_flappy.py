import argparse

from agent import Agent


def main():
    parser = argparse.ArgumentParser(description="Flappy Bird DQN")
    parser.add_argument("--train", action="store_true", help="train the agent")
    parser.add_argument("--render", action="store_true", help="show the game window")
    parser.add_argument("--episodes", type=int, default=None, help="number of episodes to train")
    parser.add_argument("--params", default="flappybirdv0", help="hyperparameter set name in parameters.yaml")
    args = parser.parse_args()

    agent = Agent(args.params)

    if args.train:
        agent.run(is_training=True, render=args.render, max_episodes=args.episodes)
    else:
        # play with the best saved model
        agent.run(is_training=False, render=True)


if __name__ == "__main__":
    main()