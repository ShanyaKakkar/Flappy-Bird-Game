import itertools
import random

import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from dqn import DQN
from experience_replay import ReplayMemory

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.MODEL_FILE = f"{param_set}.pt"

    def run(self, is_training=True, render=False, max_episodes=None):
        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
            use_lidar=False,
        )

        num_states = env.observation_space.shape[0]  # input dim
        num_actions = env.action_space.n  # output dim

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            # copy the wt & bias vals from policy => target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            best_reward = float("-inf")
        else:
            # best policy load
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()

        episodes = range(max_episodes) if max_episodes else itertools.count()

        for episode in episodes:
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_reward = 0
            terminated = False

            while not terminated and episode_reward < self.reward_threshold:
                # epsilon-greedy action selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()  # explore
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()  # exploit

                next_state, reward, terminated, _, info = env.step(action.item())

                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                episode_reward += reward.item()

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                    # train on every step once enough experiences are stored
                    if len(memory) > self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                        # sync target network with policy network
                        if steps > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            steps = 0

                state = next_state

            if is_training:
                print(f"episode={episode + 1} total reward={episode_reward:.1f} epsilon={epsilon:.4f}")

                # save best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    print(f"  new best reward {best_reward:.1f}, model saved")

                # epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            else:
                print(f"episode={episode + 1} total reward={episode_reward:.1f}")

        env.close()
        print("Training finished.")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        # target Q = r + gamma * max Q'(next_state), no future term if episode ended
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # Q-values of the actions that were actually taken
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(dim=1)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    agent = Agent("flappybirdv0")
    agent.run(is_training=True, render=False)