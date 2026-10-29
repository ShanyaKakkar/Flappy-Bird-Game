# Flappy Bird DQN

I built this project to learn reinforcement learning. It's a Deep Q-Network agent that teaches itself to play Flappy Bird using PyTorch and Gymnasium.

## What I did

- Used the `FlappyBird-v0` environment, where the bird can either flap or do nothing
- Built a DQN in PyTorch that predicts the Q-value of each action
- Used epsilon-greedy exploration, so the bird tries random moves at first and relies on the network more over time
- Added experience replay, so the agent trains on random batches of past experiences
- Added a target network to keep the training stable
- Saved the best model during training

## Files

- `agent.py`: training loop and the learning step
- `dqn.py`: the neural network
- `experience_replay.py`: replay memory
- `game_flappy.py`: run it to train or play
- `parameters.yaml`: hyperparameters

## How to run

Install the libraries:

```
pip install flappy_bird_gymnasium gymnasium torch pyyaml pygame
```

Train:

```
python game_flappy.py --train --episodes 100000
```

Watch the trained bird play:

```
python game_flappy.py
```

## What I learned

- How epsilon decay changes the balance between exploring and using what's learned. My first settings kept the bird random for too long.
- Training on every step instead of once per episode made a big difference.
- Rewards stay negative at first, and that's normal while epsilon is high.

## Next steps

- Try Double DQN
- Add a reward-vs-episode plot
- Make a web demo