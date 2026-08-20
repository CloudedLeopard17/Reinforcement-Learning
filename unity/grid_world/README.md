# DQN from Scratch — Unity ML-Agents

Deep Q-Networks implemented from scratch in PyTorch and trained on Unity ML-Agents
environments through the low-level `mlagents_envs` Python API — not `mlagents-learn`.

No RL framework is used. The replay buffer, epsilon-greedy policy, action masking,
target network, and training loop are all written here.

---

## Results

| Environment | Observation | Solved at | Greedy evaluation | Random baseline |
|---|---|---|---|---|
| Basic | 20-dim vector | ~13.6k steps | 0.93 (optimal) | ~-0.26 |
| GridWorld | 3x64x84 RGB + goal signal | 15k steps | **0.97** | -0.31 |

**Basic** is the smallest end-to-end check — a one-hot position observation and three
actions, solvable by tabular Q-learning. Its value is not the result but the
plumbing: transitions assembled correctly across the ML-Agents decision gap, rewards
attributed to the right state-action pair, timeouts distinguished from true
terminals. Any of those being wrong produces a plausible-looking but slowly-degrading
curve that is miserable to diagnose later.

**GridWorld** is the real problem, and the rest of this README is about how it failed
before it worked.

---

## GridWorld: a value function that diverged

GridWorld is goal-conditioned. The grid holds two target types; a goal-signal
observation says which to reach this episode. Touching the indicated one pays +1, the
other -1, with a 0.01 per-step cost. The same image maps to different optimal actions
depending on the signal, so the value function is Q(s, a, g), not Q(s, a) — a
universal value function approximator in the sense of Schaul et al.

With **gamma = 0.99** the agent never solved the task. With **gamma = 0.9**, and no
other change, it solves in 15,000 steps. The cause is measurable, and the measurement
is the point of this section.

### The environment has a hard ceiling on Q

The best achievable discounted return is reaching the correct target on the first
step: `-0.01 + 1 = 0.99`. That bound holds for every gamma and every policy. Any
Q-value above 0.99 is not optimism — it is provably wrong.

| | mean Q (last 1k updates) | max Q | mean discounted return | ceiling | gap |
|---|---|---|---|---|---|
| gamma = 0.9 | 0.693 | 0.782 | 0.622 | 0.99 | **+0.07** |
| gamma = 0.99 | 1.256 | 1.325 | -0.323 | 0.99 | **+1.58** |

At gamma 0.9 the value function is well calibrated — and the residual +0.07 is
partly expected, since mean Q averages over buffer states at all distances while the
discounted return is measured from episode starts, which are further from the target.

At gamma 0.99 mean Q sits **27% above a bound the environment makes impossible**, and
the loss is *lower* (0.00085 against 0.0013). The network is fitting a self-consistent
fantasy with great precision.

### The failure has a shape

The gamma 0.99 run does not simply fail to learn. Return climbs to 0.56, then
collapses to -0.51. That rise-then-collapse is what runaway value inflation looks
like: estimates grow, the greedy policy starts following inflated actions, performance
degrades, and the degraded data feeds further inflation.

An earlier gamma 0.99 run instead sat flat at chance for 50,000 steps, with 47.2%
success over the first 5k episodes and 47.9% over the last 5k — the agent reached *a*
target reliably (against a ~31% random baseline) but picked the right one no more
often than a coin flip. Same discount, two different failure modes.

### Why gamma is the lever

This is the deadly triad — function approximation, bootstrapping, off-policy learning
— and gamma is the loop gain of the feedback it creates. Each Bellman backup carries
existing error forward multiplied by gamma: contraction of 10% per backup at 0.9,
1% at 0.99.

A tabular version of this ([`maximization_bias_demo.py`](maximization_bias_demo.py))
shows that bias alone is not enough to explain the collapse. In an L-step episode
where the action never matters and the final reward is a ±1 coin flip — so true Q is
exactly 0 everywhere — the discount barely matters for short episodes:

| episode length | gamma=0.9 | gamma=0.99 | ratio |
|---|---|---|---|
| 2 | 0.194 | 0.213 | 1.1 |
| 5 | 0.190 | 0.279 | 1.5 |
| 20 | 0.033 | 0.200 | 6.1 |
| 50 | 0.001 | 0.145 | 106.7 |

GridWorld episodes are a handful of steps, where the two discounts differ by ~1.5x.
The difference is that the toy is **tabular**: every state's value is anchored
independently by the rewards it actually observes. With a neural network, an inflated
estimate at one state raises estimates at similar states through generalisation, and
those feed back through the bootstrap. Generalisation is what turns a bounded bias
into a runaway one.

A suspected accelerant, not yet measured: timeouts bootstrap by design, since
`done = not interrupted` makes a timed-out episode's target `gamma * max Q(s')` with
no reward to anchor it — a self-referential loop with gain gamma. As values inflate
the policy worsens, producing more timeouts and more unanchored self-bootstrapping.
Logging the timeout fraction against mean Q would confirm or kill this.

### A secondary effect: the action gap

The greedy policy reads only the *ordering* of Q-values, so what matters is the margin
between the best action and the runner-up against the network's approximation error.
For a decision about delay, that margin is proportional to `1 - gamma`:

    gap ≈ γ^(k-1) - γ^k = γ^(k-1) · (1 - γ)

Choosing a move that leaves the correct target k steps away versus one that leaves it
k+2 away:

| k | gap, gamma=0.9 | gap, gamma=0.99 |
|---|---|---|
| 1 | 0.207 | 0.040 |
| 3 | 0.168 | 0.039 |
| 6 | 0.122 | 0.038 |

At 0.99 navigation decisions hinge on ~0.04, small enough for ordinary approximation
error to reorder them. The value landscape flattens correspondingly: across distances
1 to 6 the spread is 0.45 at gamma 0.9 and 0.10 at 0.99.

This is a real effect but a secondary one, and it does not by itself explain the
observed failure. The correct-versus-wrong-target gap — the decision that was actually
going wrong — is 1.80 at gamma 0.9 and 1.98 at 0.99. The discount barely touches it.
Divergence, not the action gap, is the load-bearing explanation. (The action-gap
phenomenon and its error bounds are Farahmand (2011); Bellemare et al. (2016) give
operators that widen the gap deliberately. Neither is about choosing gamma — that
connection is an inference from the same principle.)

### Two hypotheses that were wrong

Recorded because eliminating them was most of the work.

**"The goal signal is swamped."** With 2 goal dimensions concatenated onto 896 conv
features, the goal supplies 0.2% of the first linear layer's input. Refuted in one
test — feeding the same images with both goal one-hots flipped the greedy action on
**64% of states**. The goal was doing real work.

**"The conditioning is too weak."** Broadcasting the goal spatially as extra input
channels, so the convolution itself is goal-aware, did produce a solve. But so did the
*original* head-concatenation architecture once the discount changed. A controlled
ablation — one variable, same seed — showed the architecture was never the cause.

### The transferable lesson

**Set gamma from the episode length, not from habit.** The effective horizon is
roughly `1 / (1 - gamma)`: 100 steps at 0.99, 10 at 0.9. GridWorld episodes are a few
steps. Discounting over a horizon an order of magnitude longer than the episode buys
nothing and turns a stable feedback loop into an unstable one. Unity's own reference
notebook uses 0.9 here.

**And log a quantity with a known bound.** Mean Q against the environment's
theoretical maximum turned a week of hypotheses into a single unambiguous number. The
return curve alone showed "not learning"; Q against the ceiling showed *why*.

---

## Implementation notes

Things that took real debugging and are easy to get wrong.

### The ML-Agents stepping model

`env.step()` takes no action and returns no observation — it advances the simulation
until some agent requests a decision. The reward and next state for an action arrive
only on that agent's *next* decision, so a `pending` dict holds the half-transition
until it can be completed. This is the main structural difference from a Gym loop and
the source of most of the subtle bugs.

- **`done = not terminal_step.interrupted`** — a time-limit truncation is not a
  terminal state and must still bootstrap.
- **`agent_id_to_index`** — row order in the observation array is not agent id and
  shifts as agents terminate and respawn. Confusing them scrambles transitions in a
  way that still trains, just badly.
- **New agent id per episode** — an agent appears in `terminal_steps` under its old id
  and in `decision_steps` under a new one on the same iteration. Popping from
  `pending` on termination and repopulating on decision handles this; the
  `if agent_id in pending` guard on both branches is what makes the gap harmless.
- **Rewards are cumulative since the agent's last decision**, not since the last
  `env.step()`. With a decision period above 1, several frames fold into one
  transition, so gamma discounts per decision rather than per frame.
- **Copy the observation** before storing it; the underlying arrays are reused.

### Action masking

GridWorld masks moves that would leave the grid, so the mask must be applied in
**two** places, not one:

```python
q      = q_net(obs, goal).masked_fill(mask, -1e9)          # action selection
q_next = target_net(next_obs, next_goal).masked_fill(next_mask, -1e9).max(1)  # target
```

Masking only at selection is the common mistake: the target's `max` then bootstraps
off an illegal action's Q-value, which is never corrected because that action is never
taken and never appears in a loss term. Those values drift freely and inflate every
target. This is why the replay buffer stores `mask_next` alongside each transition.
`-1e9` rather than `-inf` avoids NaNs if a row is ever fully masked.

Exploration samples uniformly over *legal* actions, per agent rather than per batch.
With several agents in the scene, one shared coin flip would make all of them explore
or exploit together, correlating exactly the transitions replay exists to decorrelate.

### Observation handling

- **RGB is required**, not optional. The two target types are distinguished primarily
  by colour; grayscale can collapse them and make the goal signal unusable.
- **No frame stacking.** The grid is fully observable from one image and nothing has
  momentum, so a single frame plus the goal signal is a complete state. Stacking would
  triple the buffer footprint to encode nothing.
- **The camera arrives as (3, 64, 84)** — already channels-first, so no permute is
  needed. It is also not square: 84x84 is a convention from the Atari DQN paper, not a
  requirement. Resizing would distort the aspect ratio and waste compute. A dummy
  forward pass computes the flatten size rather than hardcoding it.
- **Store uint8, divide by 255 in `forward`.** Observations arrive as float32 in
  [0, 1]; storing them that way costs 4x more than necessary.

### Goal conditioning

The `GOAL_SIGNAL` observation type is functional for `mlagents-learn` — the trainer
routes it through a conditioning mechanism, optionally a hypernetwork. For a custom
Python loop it is metadata: the array arrives in `obs` like any other and what to do
with it is a design decision. Two schemes are implemented here and both work at
gamma 0.9:

1. **Head concatenation** — concatenate the goal one-hot onto the flattened conv
   features.
2. **Input-channel broadcast** — tile the goal one-hot spatially and feed it as extra
   input channels, so the convolution itself is goal-aware from the first layer.

### Evaluating with the right sample size

Returns here are dominated by one binary outcome, so per-episode standard deviation is
`2·sqrt(p(1-p))` — maximal at p = 0.5 and shrinking towards either extreme. That makes
a mediocre agent *harder* to measure than a good one, which is backwards from where
precision is needed while debugging. The random baseline, run three times:

| episodes | measured mean | 95% CI |
|---|---|---|
| 10 | -0.252 | [-0.85, +0.35] |
| 50 | -0.308 | [-0.57, -0.04] |
| 100 | -0.387 | [-0.57, -0.21] |

All three are consistent with one underlying value, but the 10-episode interval admits
a *positive* mean return. Ten episodes cannot distinguish a random policy from one
that is learning.

At the other extreme the usual confidence interval stops applying. The final agent
averaged 0.9705 over 20 episodes, which is only reachable with zero failures — a
single wrong target caps the mean at 0.89. With zero failures the relevant bound is
the rule of three: 20 clean episodes are consistent with a failure rate up to ~14%,
100 episodes with ~3%.

---

## Repository layout

```
.
├── basic.ipynb                       # Basic — smallest end-to-end check
├── gridworld_exp.ipynb               # GridWorld — goal-conditioned DQN
└── maximization_bias_demo.py         # toy experiment behind the gamma finding
```

## Running it

The notebooks connect to the Unity Editor rather than a build:

```python
env = UnityEnvironment(file_name=None, seed=0, side_channels=[channel])
```

Run the cell, then press Play in the Editor. The Editor connection is the better
development loop anyway — you can watch the agent and change reward logic without
rebuilding. Builds only buy headless throughput and parallel environments.

```bash
conda create -n mlagents python=3.10.12 && conda activate mlagents
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install mlagents        # matched to the Unity com.unity.ml-agents package version
```

Set `time_scale` via `EngineConfigurationChannel` to speed up simulation — these
environments are simulation-bound, not compute-bound, so GPU utilisation is a poor
guide to training speed.

---

## What I would do differently

- **Log a bounded quantity from day one.** Mean Q against the environment's
  theoretical ceiling was the diagnostic that ended the investigation. It was added
  last.
- **Split metrics from the start.** Average return conflates wrong target, timeout,
  and step cost. Breaking it into success rate, mean Q, and loss was what first showed
  the agent was stuck at chance rather than slowly improving.
- **Compare like with like.** Q predicts *discounted* return; the episode logs were
  undiscounted. Until both were logged, the calibration gap could not be read off.
- **Change one thing at a time.** The first fix went in alongside an architecture
  change, and only a follow-up ablation revealed which one mattered.
- **Evaluate on a second environment instance.** `evaluate()` calls `env.reset()`
  mid-training, discarding in-flight episodes and orphaning their `pending` entries.
- **Log success rate split by goal type.** An aggregate 48% could mean 48% on each or
  95% on one and 2% on the other — different problems, indistinguishable from the
  average alone.

**Double DQN** was never added, and the measurements suggest it would not have been
the fix. At gamma 0.9 the value function is already calibrated to within 0.07, leaving
nothing for it to correct; at gamma 0.99 the estimates exceed the environment's
theoretical maximum by 27%, which is divergence rather than the bounded overestimation
Double DQN addresses.

## References

- Mnih et al. (2015), *Human-level control through deep reinforcement learning*
- van Hasselt et al. (2016), *Deep Reinforcement Learning with Double Q-learning*
- Schaul et al. (2015), *Universal Value Function Approximators*
- Farahmand (2011), *Action-Gap Phenomenon in Reinforcement Learning*
- Bellemare et al. (2016), *Increasing the Action Gap: New Operators for Reinforcement Learning*
- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, ch. 11 (the deadly triad)
