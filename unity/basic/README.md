# Basic Unity Environment DQN

The Basic environment is the smallest end-to-end Deep Q-Network check in this
repository. It uses a one-hot position observation and three actions, making it
solvable by tabular Q-learning while still exercising the Unity ML-Agents plumbing.

Its purpose is to validate that transitions are assembled correctly across the
ML-Agents decision gap, rewards are attributed to the right state-action pair, and
timeouts are distinguished from true terminal states. Bugs in any of these areas can
produce a plausible-looking but slowly degrading training curve.

## Result

| Observation | Solved at | Greedy evaluation | Random baseline |
|---|---:|---:|---:|
| 20-dim vector | ~13.6k steps | 0.93 (optimal) | ~-0.26 |

## Files

| File | Purpose |
|---|---|
| [basic.ipynb](basic.ipynb) | Trains and evaluates the DQN agent in the Basic Unity environment. |

## Running it

The notebook connects to the Unity Editor rather than a build. Follow the shared
[Unity installation and running instructions](../README.md), then open the Unity
project, run the notebook cell that creates the environment, and press **Play** in
the Editor.