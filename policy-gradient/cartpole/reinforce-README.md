# REINFORCE on CartPole: does a baseline help?

Three variants of Monte Carlo policy gradient on CartPole-v1, written from scratch in
PyTorch and compared over **10 seeds**:

1. **Plain REINFORCE** — advantage is the raw return G_t.
2. **Scalar baseline** — subtract the mean return of the current episode, G_t − mean(G).
3. **Value baseline** — subtract a learned V(s_t), G_t − V(s_t). This is the critic.

The comparison is run over seeds because a single run says almost nothing here — CartPole
is notoriously seed-sensitive, and an earlier two-run comparison gave the *opposite*
conclusion to the one below.

## Result

Tracking the *peak* 100-episode mean per seed alongside the *final* score is what makes
the differences legible — final score alone hides both the stability story and the scalar
baseline's failure.

| | mean peak | mean final | mean drop from peak | seeds retaining ≥90% of peak |
|---|---|---|---|---|
| Plain REINFORCE | 455 | 417 | 38 | 8/10 |
| Scalar baseline | 414 | 293 | 121 | 4/10 |
| Value baseline | **480** | **447** | 34 | 7/10 |

### 1. The value baseline is the clear winner

Highest peak *and* highest final score, with the smallest drop. It is the only variant
that improves on plain REINFORCE rather than trading against it, because V(s) is a genuine
state-dependent estimate of expected return — so G_t − V(s_t) is a real advantage rather
than a positional artefact.

(One honest wrinkle: on peak-*retention* it is 7/10 against plain REINFORCE's 8/10, driven
by a single seed dropping 479 → 313. At n = 10 that is within noise; the solid claims are
the peak and final advantages, not a stability win over plain REINFORCE.)

### 2. The scalar baseline does not help — and hurts stability

Worst of the three on final score, and it collapses from its peak far more often (only
4/10 seeds hold; one goes 492 → 82). An earlier single-pair run suggested it roughly
halved training time; that was a two-seed fluke plus a bug in the stopping condition, and
the 10-seed picture does not support it.

The reason is structural. With one episode per update and gamma = 1.0, the return at
timestep t is just the number of steps remaining, so subtracting the episode mean gives

    advantage(t) = (T - t) - (T + 1) / 2

which depends only on position t within the episode, not the state or action. It is a
fixed positional ramp — it removes the constant positive offset but injects a new variance
source, because it is recomputed each episode and its scale shifts as episode length
changes between updates. With no trust region, that makes destructive updates more likely.
The value baseline fixes exactly this: V(s) is a fixed target, not a per-episode constant.

### 3. All three catastrophically forget

Nearly every seed in every method reaches 500 and then falls back, sometimes hard
(500 → 103). This is the defining failure of vanilla policy gradient: nothing constrains
how far the policy moves in one update, so one unlucky batch can step off a cliff and not
recover within the run. The value baseline reduces the frequency but does not remove it —
the update step is still unbounded.

That is the cleanest possible motivation for PPO, whose clipped objective bounds exactly
this step.

## Methodology

Everything is held fixed between the three methods except the baseline: the same seed list
(policy init, NumPy, environment RNG), the same `.sum()` reduction and learning rate, and
the same greedy-evaluation stopping rule.

Two seeding details matter more than they look:

- **Seed once before the training loop, then plain `reset()` each episode.** Reseeding to
  the same seed inside the loop starts every episode from an identical state and destroys
  start-state diversity. Fixing this made all methods look worse, because they then face
  varied starts — the real task.
- **Evaluation uses its own environment instance**, so it never advances the training
  env's RNG and every method sees the same start-state stream for a given seed.

For the value baseline, one detail is load-bearing: **V(s) is detached in the policy loss**
(`advantages = returns - values.detach()`). The baseline theorem holds only if the baseline
is a constant with respect to the policy gradient; V trains separately through an MSE loss
toward the observed returns.

## What comes next

The same small environment, one change per rung, each measurable in minutes:

1. REINFORCE — no baseline, scalar baseline, value baseline. **This notebook.**
2. **A2C** — bootstrap the return instead of waiting for the full episode.
3. **PPO** — add the clipped objective that bounds the update step, addressing the
   catastrophic forgetting seen here.

## Running it

```bash
pip install "gymnasium[classic-control]" torch numpy matplotlib pandas
jupyter notebook reinforce.ipynb
```

CPU is fine — CartPole episodes are short and the networks are tiny.
