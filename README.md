# Reinforcement Learning Projects

Implementations and experiments based on *Reinforcement Learning: An Introduction*
by Sutton and Barto. The repository progresses from a small tabular control problem
to deep reinforcement learning agents running in Unity and Atari environments.

Everything is written from scratch in NumPy and PyTorch — no RL framework. The replay
buffers, epsilon-greedy policies, action masking, target networks, and training loops
are all implemented here.

## Projects

| Project | Method | Result | What it covers |
|---|---|---|---|
| [Windy Gridworld](tabular/windy-gridworld/) | SARSA | — | Exercises 6.9 and 6.10: standard actions, king moves, and stochastic wind. |
| [CartPole policy gradient](policy-gradient/cartpole/) | REINFORCE → PPO | 500.00 on 7/10 seeds | The policy-gradient ladder over 10 seeds: baseline variants of REINFORCE, then a clipped, rollout-based PPO that stops the collapse they all suffer. |
| [Unity Basic](unity/basic/) | DQN | 0.93 (optimal) | A minimal end-to-end DQN against a Unity ML-Agents environment — the check that transition handling, reward attribution, and timeout-versus-terminal logic are correct. |
| [Unity GridWorld](unity/gridworld/) | DQN | 0.97 | Goal-conditioned visual control: replay memory, action masking, target networks, and a value function that diverged. |
| [Atari Pong](atari/pong-dqn/) | DQN | 21–0 | A convolutional DQN trained from stacked, preprocessed frames, evaluated under three randomisation conditions. |

**Start with the [Unity GridWorld write-up](unity/gridworld/README.md).** It is the most
detailed of these: an agent that sat at chance for 50,000 steps with no bug in the code,
a value function that provably exceeded the environment's maximum possible return by
27%, the single measurement that caught it, and two hypotheses that turned out to be
wrong.

The [Pong evaluation](atari/pong-dqn/README.md) is the second thing worth reading — a
perfect deterministic score is weak evidence on its own, so the same checkpoint is
re-tested under sticky actions and randomised starts to separate genuine skill from a
memorised trajectory.

## Repository Layout

- `tabular/` — small, interpretable TD-control experiments.
- `policy-gradient/` — REINFORCE baseline variants and PPO on CartPole, compared
  across 10 seeds. The [write-up](policy-gradient/cartpole/README.md) is one ladder:
  each rung changes one thing and is measured against the last.
- `unity/` — DQN agents trained through the low-level ML-Agents Python API. The
  [shared Unity README](unity/README.md) covers the stepping model, action masking,
  choosing a discount factor, and evaluating with a defensible sample size.
- `atari/` — the Pong training and inference notebooks, and sample rollouts.

Trained checkpoints and TensorBoard logs are not committed; the training notebooks
produce them.

## Getting Started

The experiments are Jupyter notebooks. Create or activate a Python environment, install
the dependencies for the project you want to run, and launch Jupyter:

```bash
jupyter notebook
```

| Project | Setup |
|---|---|
| Unity (Basic, GridWorld) | [unity/README.md](unity/README.md) — ML-Agents version requirements, Editor connection, and what to do when the environment will not connect. |
| Atari Pong | [atari/pong-dqn/README.md](atari/pong-dqn/README.md) — exact dependencies and execution steps. |
| Windy Gridworld | Python, NumPy, and Matplotlib only. Details in [tabular/windy-gridworld/readme.md](tabular/windy-gridworld/readme.md). |
| CartPole policy gradient | [policy-gradient/cartpole/README.md](policy-gradient/cartpole/README.md) — one `pip install`; runs on CPU. |

## Reference

Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*,
2nd edition. [MIT Press](http://incompleteideas.net/book/the-book-2nd.html).