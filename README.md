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
| [Pong PPO](policy-gradient/pong/) | PPO | +12.31 sticky | The same PPO scaled to Atari: 8 vectorised envs, 15M steps, and a head-to-head between Monte Carlo advantage and GAE. |
| [Pendulum DDPG](policy-gradient/pendulum/) | DDPG | −113.9 over 10 seeds | The first continuous-action rung: a deterministic actor trained through its critic, and two measurement artefacts that each looked like the DDPG collapse it isn't. |
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

Those two threads meet in [PPO on Pong](policy-gradient/pong/README.md): the DQN scores 21.00
without sticky actions and 8.85 with them; PPO, trained under stickiness, scores 12.31 with
and holds a comparable score without. Which agent is "better" depends entirely on which
column you report.

## Repository Layout

- `tabular/` — small, interpretable TD-control experiments.
- `policy-gradient/` — one ladder, each rung changing one thing and measured against the
  last. [`cartpole/`](policy-gradient/cartpole/README.md) holds the REINFORCE baseline
  variants and PPO across 10 seeds; [`pong/`](policy-gradient/pong/README.md) scales the same
  PPO to Atari and compares advantage estimators at 15M steps;
  [`pendulum/`](policy-gradient/pendulum/README.md) moves to continuous actions with DDPG.
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
| Pong PPO | [policy-gradient/pong/README.md](policy-gradient/pong/README.md) — same dependencies as Atari Pong; ~2.5 h per run on a GPU. |
| Pendulum DDPG | [policy-gradient/pendulum/README.md](policy-gradient/pendulum/README.md) — same `pip install` as CartPole; runs on CPU. |

## Reference

Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*,
2nd edition. [MIT Press](http://incompleteideas.net/book/the-book-2nd.html).