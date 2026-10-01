# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Implementations and experiments based on *Reinforcement Learning: An Introduction* (Sutton &
Barto, 2nd ed.). Everything is written from scratch in NumPy and PyTorch — **no RL framework**.
Replay buffers, epsilon-greedy policies, action masking, target networks, and training loops are
all hand-implemented in the notebooks themselves, not imported from a library. When making changes,
preserve this — don't introduce `stable-baselines3`, `rllib`, etc.

All the actual work lives in Jupyter notebooks (`.ipynb`), not `.py` modules. The one exception is
`unity/gridworld/maximization_bias_demo.py`, a standalone tabular script. There is no package to
install, no build step, no test suite, and no linter configured — this is a research/writeup repo,
not a library.

## Repository layout

- `tabular/windy-gridworld/` — SARSA on Windy Gridworld (exercises 6.9/6.10). Deps: NumPy,
  Matplotlib only.
- `policy-gradient/cartpole/` — the policy-gradient ladder on CartPole-v1, compared across 10
  seeds. `reinforce.ipynb` holds the baseline variants (none / scalar / value);
  `ppo_rollout.ipynb` adds the clipped objective and fixed-length 2048-step rollouts. Both are
  covered by the one directory README. Evaluate a stochastic policy greedily — PPO's entropy
  bonus makes the training curve understate the policy by up to 340 points on a seed whose
  greedy score is a perfect 500.
- `unity/` — DQN agents trained through the **low-level `mlagents_envs` Python API**, not
  `mlagents-learn`. `unity/README.md` is the shared reference for the ML-Agents stepping model,
  action masking, and evaluation methodology — read it before touching anything under `unity/`.
  - `unity/basic/` — minimal end-to-end DQN sanity check (20-dim vector obs).
  - `unity/gridworld/` — goal-conditioned visual DQN (3x64x84 RGB + goal signal). Its README
    documents a real debugging investigation (a value function that diverged 27% above the
    environment's provable ceiling) — read it before changing gamma, the network architecture, or
    the goal-conditioning scheme.
- `policy-gradient/pong/` — PPO on Atari Pong, the top rung of the same ladder. Uses ALE's
  **native** vectorised env (`gym.make_vec` with no `vectorization_mode`, ~2x the async
  Python-wrapper path); `RecordEpisodeStatistics` must sit *inside* `ClipReward` or every
  reported score is silently wrong; autoreset is `NEXT_STEP`, so the row after a termination
  is fabricated and is dropped via a `valid` mask. `dones` records `terminated` (a frame-cap
  truncation must still bootstrap) while the autoreset mask needs `terminated | truncated` —
  two flags, two jobs, same loop. 15M steps ≈ 2.5 h per run, so multi-seed comparison is out
  and the statistical weight moves to the evaluation protocol.
- `policy-gradient/pendulum/` — DDPG on Pendulum-v1, 10 seeds; the first continuous-action
  project. The actor loss must be `-critic(s, actor(s))` — feeding the critic buffer actions severs
  the path to the actor, leaves `actor.grad` as `None`, and Adam silently skips the step. Pendulum
  never terminates (every episode ends by truncation), so storing `done or truncated` as terminal
  corrupts every episode boundary. Q is bounded to [-1409, 0] at gamma=0.99: any positive Q is
  provably wrong. Results saved to `ddpg_pendulum.npz`.
- `policy-gradient/mujoco/hopper/` — DDPG vs TD3 on Hopper-v5, 5 seeds x 1M steps, both arms from
  one `run(..., use_td3)`. Hopper observations are **float64** (Pendulum's were float32), so
  `torch.tensor(state)` needs an explicit dtype or the first forward pass dies. `max_action` is 1.0
  here. Writing the TD3 arm produced four bugs that all ran without erroring: `target` assigned only
  on the DDPG branch; the second critic missing from the optimizer; both critics built with the same
  seed, making `min(Q1, Q2)` identically `Q1` so TD3 silently reduced to DDPG; and target smoothing
  applied to the exploration action instead of the target action. Seeds run as forked CPU processes
  (CPU measured faster than CUDA for 256x256 nets at batch 256, and fork cannot re-init a CUDA
  context). ~10 h per arm: `random.sample` on a `deque` is O(n) and costs 14.1 ms per batch at 1M
  entries vs 0.49 ms pre-allocated.
- `atari/pong-dqn/` — CNN DQN (Mnih et al. Nature DQN architecture) on Atari Pong from stacked
  frames.

Each project directory has its own README with exact `pip install` commands and run steps — check
it before assuming a dependency set. Nothing is committed for trained artifacts: `*.pth`
checkpoints, `logs_*/`, `runs/`, `.ipynb_checkpoints/`, and `__pycache__/` are all gitignored;
running the training notebooks regenerates them.

## Running things

There's no single entry point — launch Jupyter and run the relevant notebook's cells top to
bottom:

```bash
jupyter notebook
```

Per-project setup:

| Project | Install |
|---|---|
| Windy Gridworld | `numpy`, `matplotlib` |
| CartPole REINFORCE | `pip install "gymnasium[classic-control]" torch numpy matplotlib pandas` |
| Atari Pong | `pip install gymnasium "gymnasium[atari,accept-rom-license]" ale-py opencv-python imageio torch matplotlib pandas tensorboard` |
| Unity (Basic, GridWorld) | `conda create -n mlagents python=3.10.12 && conda activate mlagents && pip install torch --index-url https://download.pytorch.org/whl/cu121 && pip install mlagents` — the `mlagents` pip version must match the Unity `com.unity.ml-agents` package version, or you get an explicit API-incompatibility error. |

Unity notebooks connect to the **Unity Editor**, not a build: run the cell that creates
`UnityEnvironment(file_name=None, ...)`, then press Play in the Editor. If it times out after 60s
(`UnityTimeOutException`), check in order: wrong scene at index 0 in Build Settings, Behavior Type
not `Default`, a stale process holding the port (`pkill -f <BuildName>`), aggressive IL2CPP
stripping removing gRPC types, or a headless machine needing `no_graphics=True`/`xvfb-run`.

Pong training writes TensorBoard logs: `tensorboard --logdir logs_pong_dqn_frame_skips`.

## Architectural knowledge that isn't obvious from one file

### The ML-Agents stepping model (unity/)

`env.step()` takes no action and returns no observation — it advances the sim until *some* agent
needs a decision. Reward and next-state for an action only arrive on that agent's *next* decision,
so the training loop keeps a `pending` dict holding the half-finished transition. The loop shape:

```python
decision_steps, terminal_steps = env.get_steps(behavior_name)
# 1. agents whose episode just ended  -> complete and store their transition
# 2. agents needing an action         -> complete the previous transition, then act
env.step()
```

Sharp edges, all covered in `unity/README.md`:
- `done = not terminal_step.interrupted` — a timeout is not terminal and must still bootstrap.
- `agent_id_to_index` row order shifts as agents terminate/respawn; don't assume it's stable.
- An agent gets a *new* id each episode — it appears in `terminal_steps` under the old id and
  `decision_steps` under the new one in the same iteration. The `if agent_id in pending` guard on
  both branches is what makes this safe.
- Rewards are cumulative since the agent's *last decision*, not since the last `env.step()` — with
  decision period > 1, gamma discounts per decision, not per frame.
- Copy observations before storing — the underlying arrays are reused by the API.
- Visual observations arrive channels-first (e.g. `(3, 64, 84)`), already `[0,1]` float32 — store
  `uint8` in the replay buffer and divide by 255 inside `forward` (4x buffer memory otherwise).
  They aren't necessarily square; compute the flatten size via a dummy forward pass rather than
  hardcoding it.
- Select observations by shape/`observation_type`, never by fixed index — sensor order isn't
  guaranteed stable.

### Action masking (unity/gridworld)

Illegal-action masks must be applied in **both** places or Q-values for illegal actions drift
unboundedly (they're never selected, so never corrected by a loss term):

```python
q      = q_net(obs).masked_fill(mask, -1e9)                      # action selection
q_next = target_net(next_obs).masked_fill(next_mask, -1e9).max(1) # target — easy to forget
```

`True` means the action is unavailable. `-1e9` rather than `-inf` avoids NaNs on a fully-masked
row. The replay buffer must store `mask_next` alongside each transition to make this possible.
Exploration samples uniformly over legal actions *per agent*, not per batch, so multiple agents in
one scene don't all explore/exploit in lockstep.

### Choosing gamma

Set it from episode length, not habit: effective horizon ≈ `1 / (1 - gamma)`. On GridWorld
(episodes a handful of steps), gamma=0.99 produced a value function 27% above the environment's
provable ceiling and a policy that rose then collapsed; gamma=0.9 solved it in 15k steps with no
other change. This is a case of the deadly triad (function approximation + bootstrapping +
off-policy) — see `unity/gridworld/README.md` for the full investigation and
`unity/gridworld/maximization_bias_demo.py` for the tabular ablation isolating discount from
generalization. General habit worth carrying into new experiments: log a quantity with a known
bound (e.g. mean Q vs. the environment's max achievable return) rather than relying on the return
curve alone. On Pong with clipped rewards that bound is sharp and cheap: gamma=0.99 gives
`|V| <= 1/(1-gamma) = 100`, but points are >=35 agent steps apart and a game ends at 21, so
`V <= sum (0.99^35)^i = 2.37`. A critic drifting past 2.4 is provably broken — a `< 100` check
would never notice.

### Evaluation sample sizes

Binary-outcome-dominated returns have per-episode std `2·sqrt(p(1-p))` — maximal at p=0.5, so a
mediocre policy is *harder* to measure precisely than a good or bad one. For near-perfect policies
(zero observed failures), the normal CI is invalid; use the rule of three instead (95% upper bound
on failure rate ≈ `3/n`). Don't report a headline success rate from fewer than ~100 episodes.

When the start state drives most of the return variance (Pendulum, Pong without sticky actions),
evaluate from **fixed** start seeds so checkpoints are paired — on Pendulum this shrank a
window-comparison CI from ±19.2 to ±1.0 with nothing else changed. Keep per-episode returns, not
just the mean.

Don't detect forgetting by comparing a seed's *peak* rolling mean with its final window: the max
of many overlapping noisy windows is biased upward even when nothing changes. On Pendulum it
reported a 52-point "collapse" where a zero-forgetting bootstrap predicts 43.6 from selection
alone. It was valid for the REINFORCE comparison only because those drops (500 → 82) dwarfed the
noise. Compare two fixed windows instead.

Related trap, from Hopper: a swing expressed as a multiple of the arm's *own* evaluation noise is a
within-arm test of "is this movement real", not a cross-arm stability metric. TD3 swung 206 points
against DDPG's 548 yet scored a *higher* multiple (8.9x vs 3.6x), because a better policy scores
consistently across fixed starts and shrinks the denominator. Across arms, compare absolute swings.

Where no bound on the value function exists (MuJoCo), measure overestimation directly: load the
saved actor and critic, roll out from the fixed eval starts, and compare the critic's estimate
against the discounted return actually obtained. On Hopper this gave 1.35x for DDPG and 0.99x for
TD3 — a critic that is merely optimistic, which a provable ceiling would never catch.

### REINFORCE baseline comparisons (policy-gradient/cartpole)

When comparing training variants, hold everything fixed except the one thing under test (same
seed list, same reduction, same LR, same stopping rule), seed once before the training loop rather
than every episode (reseeding every episode kills start-state diversity and makes results look
better than they are), and evaluate with a separate environment instance so evaluation doesn't
perturb the training env's RNG stream. For a learned value baseline, it must be detached in the
policy loss (`returns - values.detach()`) — the baseline theorem only holds if it's constant with
respect to the policy gradient — *and* the value network's parameters must be passed to the
optimizer alongside the policy's (`Adam(list(policy.parameters()) + list(value_net.parameters()))`).
Omitting them is silent: the MSE loss still backprops, nothing errors, and the run still trains,
but V stays frozen at its random initialisation for the whole run.
