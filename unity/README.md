# Unity Reinforcement Learning

These projects implement Deep Q-Networks from scratch in PyTorch and train them
through the low-level Unity ML-Agents Python API.

## Solved Environments

| Environment | Observation | Result |
|---|---|---|
| [Basic](basic/) | 20-dimensional vector | Solved in about 13.6k steps; greedy evaluation reached 0.93 (optimal). |
| [GridWorld](grid_world/) | 3x64x84 RGB image plus goal signal | Solved in about 15k steps; greedy evaluation reached 0.97. |

Basic is the small end-to-end environment for checking transition handling,
reward attribution, and terminal versus timeout behavior. GridWorld extends the
problem to goal-conditioned visual control with replay memory, action masking,
and target networks.

## Installation

Create the environment and install the Python dependencies:

```bash
conda create -n mlagents python=3.10.12
conda activate mlagents
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install mlagents
```

The `mlagents` package version must match the Unity `com.unity.ml-agents` package.

## Running

Both notebooks connect to the Unity Editor rather than a build. Open the relevant
Unity project, run the notebook cell that creates the environment, and press
**Play** in the Editor.

- [Unity environment introduction](unity_env_intro.ipynb) covers the ML-Agents
	Python API, behaviors, `BehaviorSpec`, decision and terminal steps, action
	submission, and the Unity-to-Gym wrapper.
- [Basic environment notebook](basic/basic.ipynb)
- [GridWorld notebook](gridworld/gridworld_exp.ipynb)