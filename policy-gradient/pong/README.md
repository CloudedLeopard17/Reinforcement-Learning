# PPO on Pong: the policy-gradient ladder at Atari scale

The [CartPole notebooks](../cartpole/README.md) ended on a claim — PPO's clipped objective
bounds the update step, so a policy that reaches a good score stays there. That was measured
where an episode is 500 steps and a run takes minutes. This is the same algorithm on Atari:
episodes of ~3,800 agent steps, 4x84x84 pixel observations, 15M steps (**60M frames**) per
run at ~2.5 hours each.

Two things change at this scale, and the notebook is built around both: **vectorised
environments**, because one env cannot feed a conv net, and **GAE**, because a 128-step
rollout is now 1/30th of an episode rather than a whole one. MC advantage versus GAE is the
notebook's main experiment, run head-to-head at matched seed and budget.

## Result

| | training score | greedy, sticky 0.25 | greedy, no stickiness |
|---|---|---|---|
| MC advantage | +8.30 | **+12.31 ± 0.81** | +11.12 ± 3.73 |
| GAE λ=0.95 | +9.07 | +10.84 ± 1.48 | **+16.41 ± 2.17** |

Greedy evaluation, 32 episodes each, 95% CI.

### 1. GAE is faster early, and the advantage shrinks

Steps to first reach a given training score:

| score | MC | GAE | speedup |
|---|---|---|---|
| −20 | 1.64M | 1.02M | 1.60× |
| −15 | 3.69M | 2.36M | 1.57× |
| −10 | 5.22M | 4.71M | 1.11× |
| 0 | 9.52M | 7.99M | 1.19× |
| +8 | 14.64M | 12.80M | 1.14× |

~1.5× while the policy is still bad, settling to ~1.15× once both are playing. The honest
claim is 15–20% fewer steps to a given score — quoting the early ratio would overstate it by
a third.

Why it helps at all: the weight GAE puts on a TD error *l* steps ahead is (γλ)^l, so the
credit-assignment window is `1/(1 − γλ) = 16.8` steps against γ alone's `1/(1 − γ) = 100`.
A Pong point takes ~35 steps, so the MC advantage is dominated by *when the next point
lands* — a property of the rally, not of the action taken. GAE keeps credit inside about half
a rally.

### 2. The two estimators produce differently-shaped policies

This is the more interesting result, and it is not about speed. Under sticky actions the two
models are indistinguishable (−1.47 ± 1.69). Remove stickiness and GAE is 5.3 points better
(+5.29 ± 4.32, significant) — because the **MC model does not improve at all** when the noise
is removed (12.31 → 11.12) while the **GAE model gains 5.6 points** (10.84 → 16.41).

So the MC policy is insensitive to action-repeat noise and the GAE policy is not. The shorter
credit-assignment window buys a precision that sticky actions then take away. Both arms saw
identical data distributions, so this is a property of the estimator rather than the task.

### 3. Against the DQN on the same game

| | sticky 0.25 | no stickiness |
|---|---|---|
| [DQN](../../atari/pong-dqn/README.md) (trained deterministic) | 8.85 | **21.00** |
| PPO + MC (trained sticky, 15M) | **12.31** | 11.12 |
| PPO + GAE (trained sticky, 15M) | 10.84 | 16.41 |

The DQN wins on the condition it trained under and collapses off it — 21.00 → 8.85, which is
exactly what its inference notebook was built to detect. PPO holds across both conditions but
is well short of 21, and its training curve was still climbing at 15M: budget-limited, not
converged.

Training on the harder condition is what makes this table possible. Sticky→deterministic
transfers; deterministic→sticky does not. One 2.5-hour run therefore yields both columns,
whereas training without stickiness would have needed a second run to get the first column —
and might have produced the DQN's 8.85.

## Things that are easy to get wrong

**Wrapper order.** `RecordEpisodeStatistics` must sit *inside* `ClipReward`, so it accumulates
raw ±1-per-point rewards while the agent sees clipped ones. Reversed, the agent trains
identically and every reported score is silently halved-or-worse. The native env also clips
in C++ by default, hence `reward_clipping=False`.

**Frameskip, in both directions.** With the native `AtariVectorEnv`, pass nothing — it skips 4
internally, and `frameskip=1` would disable skipping entirely. On the Python-wrapper path the
opposite holds: `gym.make(..., frameskip=1)` is required or `AtariPreprocessing` skips 4 on
top of the env's 4. The wrapper raises if you forget; the native env does not.

**`terminated` and `terminated | truncated` are different flags doing different jobs**, in the
same loop. An episode cut at the frame cap is not terminal, so its return must bootstrap
through the cut — that is `dones`. But the vector env autoresets after either, so the
autoreset bookkeeping needs both. (Same distinction as `done = not terminal_step.interrupted`
in the [Unity notebooks](../../unity/README.md).)

**`NEXT_STEP` autoreset fabricates one row per episode.** The step after a termination ignores
your action and returns the reset observation with reward 0 and `terminated=False`. Stored
naively, that pairs a terminal observation with an action that had no effect and gives it the
*next* episode's discounted return as a value target. A `valid` mask keeps those rows out of
the loss while leaving them in the return recursion, which is correct — the mask at the real
terminal already stops the recursion. Measured cost before fixing: 1 row in 30,720 with a
trained policy, 1 in 1,200 with a random one. It explains none of the results above; it is
fixed for correctness, not for score.

**Greedy evaluation is blind early.** A freshly initialised network's argmax picks the same
action on all 512 real Pong states tested, while its sampled policy is essentially uniform
(entropy 1.7916 against a maximum of 1.7918). A constant action loses 21–0, so greedy
evaluation reports exactly −21.00 ± 0.00 — here for 1.2M steps after the training score had
already begun to move. It is the CartPole training-vs-greedy gap running the other way, and
it nearly cost a run that was diagnosed as broken when it was merely slow.

## Evaluation methodology

32 episodes, 8 per env across 4 envs — a fixed count *per env*, not the first 32 to finish,
which would oversample short episodes, and short means lost quickly.

At the measured per-episode sd of ~2.3 under sticky actions, 32 episodes resolves ±1.3 points;
separating two checkpoints by 1 point needs ~200. The non-sticky condition is the *weaker*
measurement despite being the easier task: removing sticky actions removes the last per-step
randomness, so with an argmax policy the outcome is decided by the no-op start and the sd
rises to ~6–11.

Both estimator arms are single runs. At 2.5 hours each the 10-seed protocol from the CartPole
notebooks is not available, so the speedup claim rests on its consistency across thresholds,
and the estimator-shape result in §2 wants more episodes before it is leaned on.

## Configuration

Follows the PPO paper's Atari table, with two deviations noted below.

| | |
|---|---|
| envs / horizon | 8 × 128 = 1024 per rollout |
| minibatch / epochs | 256 × 4 epochs = 16 updates per rollout |
| γ / λ | 0.99 / 0.95 |
| learning rate | 2.5e-4 × α, annealed to 0 |
| clip ε | 0.1 × α, annealed to 0 |
| entropy / VF coefficient | 0.01 / 1.0 |
| budget | 15M agent steps = 60M frames |

Deviations: 4 epochs rather than the paper's 3, and no gradient-norm clipping — both arms run
without it, so the comparison is unaffected. The budget is 15M steps against the paper's 10M
(40M frames), the extra being for the sticky actions the paper predates.

## Running it

```bash
pip install "gymnasium[atari]" ale-py torch numpy matplotlib pandas tensorboard
jupyter notebook ppo.ipynb
```

Roughly 2.5 hours per arm on one GPU at ~1,500–2,000 agent-steps/s. Checkpoints
(`ppo_pong_15M.pth`, `ppo_pong_gae_15M.pth`) and TensorBoard event files are not committed —
the notebook regenerates them.

```bash
tensorboard --logdir .
```
