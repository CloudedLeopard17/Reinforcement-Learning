# Unity ML-Agents

Deep Q-Networks written from scratch in PyTorch and trained through the low-level
`mlagents_envs` Python API — not `mlagents-learn`. The replay buffer, epsilon-greedy
policy, action masking, target network, and training loop are all implemented here.

This file covers what is true across every environment in this directory. Each
environment folder has its own README for results and anything specific to it.

## Solved environments

| Environment | Observation | Solved at | Greedy evaluation | Random baseline |
|---|---|---:|---:|---:|
| [Basic](basic/) | 20-dim vector | ~13.6k steps | 0.93 (optimal) | ~-0.26 |
| [GridWorld](gridworld/) | 3x64x84 RGB + goal signal | ~15k steps | **0.97** | -0.31 |

**Basic** is the smallest end-to-end check — solvable by tabular Q-learning, so its
value is the plumbing rather than the result.

**GridWorld** is goal-conditioned visual control, and the more interesting write-up:
it includes a value function that provably diverged, the measurement that caught it,
and two hypotheses that turned out to be wrong. See
[unity/grid_world/README.md](gridworld/README.md).

## Notebooks

| Notebook | Purpose |
|---|---|
| [unity_env_intro.ipynb](unity_env_intro.ipynb) | Connecting to a Unity environment, reading behavior specs and observations, stepping the simulation. |
| [basic/basic.ipynb](basic/basic.ipynb) | DQN on the Basic environment. |
| [gridworld/gridworld_exp.ipynb](gridworld/gridworld_exp.ipynb) | Goal-conditioned DQN on GridWorld, with the full debugging write-up in markdown. |

---

## The ML-Agents stepping model

This is the main structural difference from a Gym loop and the source of most of the
subtle bugs.

`env.step()` takes no action and returns no observation — it advances the simulation
until some agent requests a decision. The reward and next state for an action arrive
only on that agent's *next* decision, so a `pending` dict holds the half-transition
until it can be completed. The loop is therefore inside-out compared to Gym:

```python
decision_steps, terminal_steps = env.get_steps(behavior_name)
# 1. agents whose episode just ended  -> complete and store their transition
# 2. agents needing an action         -> complete the previous transition, then act
env.step()
```

Details that are easy to get wrong:

- **`done = not terminal_step.interrupted`** — a time-limit truncation is not a
  terminal state and must still bootstrap. Treating timeouts as terminals biases every
  value estimate downward.
- **`agent_id_to_index`** — row order in the observation array is not agent id, and it
  shifts as agents terminate and respawn. Confusing them scrambles transitions in a way
  that still trains, just badly.
- **New agent id per episode** — an agent appears in `terminal_steps` under its old id
  and in `decision_steps` under a new one on the same iteration. Popping from `pending`
  on termination and repopulating on decision handles this; the
  `if agent_id in pending` guard on both branches is what makes the gap harmless, since
  the first decision of an episode has no prior action to complete.
- **Rewards are cumulative since the agent's last decision**, not since the last
  `env.step()`. With a decision period above 1, several frames fold into one transition,
  so gamma discounts per decision rather than per frame.
- **Copy the observation** before storing it; the underlying arrays are reused.
- **Several agents usually run in parallel.** One `env.step()` yields a batch of
  transitions, so anything counted per transition — replay ratio, epsilon decay — is
  silently scaled by the agent count. Print `len(decision_steps)` before reasoning about
  throughput.

## Reading observation specs

Never index observations by position. The order follows the sensor order on the Unity
agent and changes if anyone reorders components. Select by shape or by
`observation_type`:

```python
for i, s in enumerate(spec.observation_specs):
    print(i, s.name, s.shape, s.observation_type)
```

- Visual observations arrive **channels-first**, e.g. `(3, 64, 84)` — no permute needed
  before `Conv2d`, unlike Gym/Atari which is NHWC.
- They are not necessarily square. 84x84 is a convention from the Atari DQN paper, not a
  requirement; resizing distorts the aspect ratio and wastes compute. A dummy forward
  pass should compute the flatten size rather than hardcoding it.
- They arrive as float32 in `[0, 1]`. Store `uint8` in the replay buffer and divide by
  255 inside `forward` — otherwise the buffer costs 4x more than it needs to.
- An `ObservationType.GOAL_SIGNAL` sensor describes *what the agent is being asked to
  do*, not what it is seeing. For `mlagents-learn` that tag is functional; for a custom
  Python loop it is metadata, and how to condition on it is a design decision.

## Action masking

Environments that mask illegal actions (GridWorld masks moves that leave the grid)
require the mask in **two** places, not one:

```python
q      = q_net(obs).masked_fill(mask, -1e9)                      # action selection
q_next = target_net(next_obs).masked_fill(next_mask, -1e9).max(1) # target
```

Masking only at selection is the common mistake: the target's `max` then bootstraps off
an illegal action's Q-value, which is never corrected because that action is never taken
and never appears in a loss term. Those values drift freely and inflate every target.
This is why the replay buffer must store `mask_next` alongside each transition.

`-1e9` rather than `-inf` avoids NaNs if a row is ever fully masked.

Note the polarity: **`True` means the action is unavailable.**

Exploration samples uniformly over *legal* actions, per agent rather than per batch.
With several agents in the scene, one shared coin flip would make all of them explore or
exploit together, correlating exactly the transitions replay exists to decorrelate.

## Choosing the discount factor

Set gamma from the episode length, not from habit. The effective horizon is roughly
`1 / (1 - gamma)` — 100 steps at 0.99, 10 steps at 0.9.

On GridWorld, where episodes last a handful of steps, gamma 0.99 produced a value
function that **exceeded the environment's theoretical maximum by 27%** and a policy that
climbed to 0.56 before collapsing to -0.51. Gamma 0.9, with no other change, solved it in
15k steps. The full measurement is in the
[GridWorld write-up](gridworld/README.md#the-environment-has-a-hard-ceiling-on-q).

The transferable habit: **log a quantity with a known bound.** Mean Q against the
environment's best achievable return turned a long run of competing hypotheses into a
single unambiguous number. A return curve alone shows "not learning"; Q against the
ceiling shows *why*.

## Evaluating with the right sample size

Returns in these environments are dominated by one binary outcome, so per-episode
standard deviation is `2·sqrt(p(1-p))` — maximal at p = 0.5 and shrinking towards either
extreme. A mediocre agent is therefore *harder* to measure than a good one, which is
backwards from where precision is needed while debugging.

The GridWorld random baseline, measured at several sample sizes:

| episodes | measured mean | 95% CI |
|---|---|---|
| 10 | -0.252 | [-0.85, +0.35] |
| 50 | -0.308 | [-0.57, -0.04] |
| 100 | -0.387 | [-0.57, -0.21] |

All are consistent with one underlying value, but the 10-episode interval admits a
*positive* mean return — ten episodes cannot distinguish a random policy from one that is
learning.

At the other extreme the usual `mean ± 1.96·SE` interval stops applying, because it needs
enough of both outcomes in the sample. A near-perfect agent produces zero failures, and
the normal approximation then returns an interval extending above the maximum possible
return. The right tool is a binomial bound: with zero failures in *n* episodes the 95%
upper bound on the failure rate is roughly `3/n`.

| result | 95% upper bound on failure rate |
|---|---|
| 0 failures in 20 | 13.9% |
| 0 failures in 50 | 5.8% |
| 0 failures in 100 | 3.0% |

So a clean 20-episode run is still consistent with a policy that fails one time in seven.
100 episodes is the honest sample size for a headline number.

---

## Installation

```bash
conda create -n mlagents python=3.10.12
conda activate mlagents
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install mlagents
```

The `mlagents` Python package version must match the Unity `com.unity.ml-agents` package
version. A mismatch usually surfaces as an explicit API-incompatibility error rather than
a timeout.

## Running

The notebooks connect to the Unity **Editor** rather than a build:

```python
env = UnityEnvironment(file_name=None, seed=0, side_channels=[channel])
```

Run the cell, then press **Play** in the Editor. The Editor connection is the better
development loop — you can watch the agent and change reward logic without rebuilding.
Builds only buy headless throughput and parallel environments.

Set `time_scale` via `EngineConfigurationChannel` to speed up simulation. These
environments are simulation-bound rather than compute-bound, so GPU utilisation is a poor
guide to training speed; the levers that matter are `time_scale`, `no_graphics`, and the
number of agents in the scene.

### If the environment will not connect

A `UnityTimeOutException` after 60 seconds means Python waited and nothing handshook on
the gRPC port. In rough order of likelihood:

1. **Wrong scene in the build.** Unity launches whatever sits at index 0 in
   File → Build Settings → Scenes In Build, regardless of which scene was open when you
   clicked Build.
2. **Behavior Type is not `Default`.** `Heuristic Only` or `Inference Only` means the
   agent never opens a communicator. Check that `Model` is `None` too.
3. **A stale process is holding the port.** `pkill -f <BuildName>`, or pass a different
   `base_port`.
4. **Player Settings.** For builds, use the Mono scripting backend with managed stripping
   set to Minimal — aggressive stripping under IL2CPP removes protobuf/gRPC types the
   communicator needs, and the failure is silent.
5. **No display.** On a headless machine, pass `no_graphics=True` or build as a
   Dedicated Server. Note that `no_graphics` blanks camera sensors, so use a server build
   or `xvfb-run` for visual observations.

Running the binary by hand with `-logFile -` shows the real error; the Python-side
exception never does.