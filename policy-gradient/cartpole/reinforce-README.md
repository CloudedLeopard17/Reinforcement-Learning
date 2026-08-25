# REINFORCE on CartPole: does a baseline help?

Vanilla REINFORCE (Monte Carlo policy gradient) on CartPole-v1, compared against
REINFORCE with the simplest possible baseline — subtracting the mean return of the
current episode from each timestep's return. Written from scratch in PyTorch: the
policy network, the backward return accumulation, and both training loops.

The comparison is run over **10 seeds**. A single run of either method says almost
nothing here — CartPole is notoriously seed-sensitive, and an earlier two-run
comparison gave the *opposite* conclusion to the one below.

## Result

| | REINFORCE | REINFORCE + scalar baseline |
|---|---|---|
| median episodes to solve | ~220 (9/10 solved) | ~265 (10/10 solved) |
| mean peak score | 455 | 414 |
| mean final score | 417 | 293 |
| mean drop from peak | 38 | 121 |
| seeds retaining ≥90% of peak | 8/10 | 4/10 |

Three findings, each measured rather than asserted.

### 1. The scalar baseline does not speed up learning

Median episodes-to-solve is comparable between the two methods. An earlier single-pair
run suggested the baseline roughly halved training time; that was a two-seed fluke plus
a bug in the stopping condition, and the 10-seed picture does not support it.

The reason is structural. With one episode per update and gamma = 1.0, the return at
timestep t is just the number of steps remaining, so subtracting the episode mean gives

    advantage(t) = (T - t) - (T + 1) / 2

which depends only on the position t within the episode, not on the state or the action.
It is a fixed positional ramp — the same for every episode of a given length. It
correlates loosely with state value (early = upright, late = about to fall) but carries
no direct information about which action was good.

### 2. The baseline hurts stability

This only shows up once you track the *peak* score per seed alongside the final score.
Both methods reach similar peaks, but plain REINFORCE mostly holds it (8/10 seeds end
near their best) while the baseline version collapses far more often (only 4/10 hold;
one seed peaks at 492 and ends at 82).

A baseline is meant to *reduce* variance, so this needs explaining. The likely mechanism
— a hypothesis at n = 10, not a proven claim — is that the scalar baseline is recomputed
each episode from that episode's own returns, so as episode length swings between
updates the advantage scale shifts with it. It removes the constant positive offset but
injects a new source of update-to-update variance, and with no trust region to bound the
step, that makes destructive updates more likely.

This is about the *scalar* baseline specifically, not baselines in general. A learned
value function V(s) is a fixed, state-dependent target rather than a per-episode
positional constant — which is exactly why it is the natural next step.

### 3. Both methods catastrophically forget

Nearly every seed in both methods reaches 500 and then falls back, sometimes hard
(500 → 103, 500 → 151). This is the defining failure of vanilla policy gradient: nothing
constrains how far the policy moves in one update, so a single unlucky batch of
high-return episodes can step off a cliff, and the policy does not recover within the
run.

That is the cleanest possible motivation for PPO, whose clipped objective exists to bound
exactly this step.

## Methodology

Everything is held fixed between the two methods except the baseline: the same seed list
(applied to policy init, NumPy, and the environment RNG), the same `.sum()` reduction and
learning rate, and the same stopping rule (greedy evaluation over 30 episodes clearing
450).

Two seeding details matter more than they look:

- **Seed once before the training loop, then plain `reset()` each episode.** Reseeding to
  the same seed *inside* the loop starts every episode from an identical state and
  destroys start-state diversity. An earlier version did this; fixing it made both
  methods look worse, because they then face varied starts — the real task.
- **Evaluation uses its own environment instance**, so it never advances the training
  env's RNG and both methods see the same start-state stream for a given seed.

## What comes next

This is the first rung of a ladder built on the same small environment, so each change is
measurable in minutes:

1. REINFORCE, and REINFORCE + scalar baseline — this notebook.
2. **Learned value baseline** — replace the positional ramp with V(s), turning the
   baseline into a genuine advantage. This is the critic.
3. **A2C** — bootstrap the return instead of waiting for the full episode.
4. **PPO** — add the clipped objective that bounds the update step, addressing the
   catastrophic forgetting seen here.

## Running it

```bash
pip install "gymnasium[classic-control]" torch numpy matplotlib
jupyter notebook reinforce.ipynb
```

CPU is fine — CartPole episodes are short and the networks are tiny.
