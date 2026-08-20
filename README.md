# Reinforcement Learning Projects

Implementations and experiments based on *Reinforcement Learning: An Introduction*
by Sutton and Barto. The repository progresses from a small tabular control problem
to deep reinforcement learning agents running in Unity and Atari environments.

## Projects

| Project | Method | What it covers |
|---|---|---|
| [Windy Gridworld](tabular/Windy%20Gridworld/) | SARSA | Exercises 6.9 and 6.10: standard actions, king moves, and stochastic wind. |
| [Unity Basic](unity/basic/basic.ipynb) | DQN | A minimal end-to-end DQN experiment against a Unity ML-Agents environment. |
| [Unity GridWorld](unity/grid_world/) | DQN | Goal-conditioned visual control, replay memory, action masking, target networks, and debugging value-function divergence. |
| [Atari Pong](atari/PONG%20DQN/) | DQN | A convolutional DQN trained from stacked, preprocessed Pong frames, with separate evaluation experiments. |

## Repository Layout

- `tabular/` contains small, interpretable TD-control experiments.
- `unity/` contains notebooks for connecting to Unity ML-Agents environments and
  training DQN agents from scratch with PyTorch.
- `atari/` contains the Pong training and inference notebooks, a trained checkpoint,
  TensorBoard logs, and sample rollouts.

## Getting Started

The experiments are Jupyter notebooks. Create or activate a Python environment,
install the dependencies required by the project you want to run, and launch
Jupyter:

```bash
jupyter notebook
```

For the exact Atari dependencies and execution steps, see the
[Pong DQN README](atari/PONG%20DQN/README.md). For Unity GridWorld setup, including
the Unity Editor connection and ML-Agents version requirements, see the
[Unity GridWorld README](unity/grid_world/README.md). The tabular project only
requires Python, NumPy, and Matplotlib; its details are in the
[Windy Gridworld README](tabular/Windy%20Gridworld/readme.md).

## Reference

Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*,
2nd edition. [MIT Press](http://incompleteideas.net/book/the-book-2nd.html).
