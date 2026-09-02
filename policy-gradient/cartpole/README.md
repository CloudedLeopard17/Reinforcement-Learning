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
| Value baseline | **477** | **458** | **19** | **9/10** |

### 1. The value baseline is the clear winner

It leads on every column: highest peak, highest final score, half the drop from peak, and
the best retention. It is also the only variant to solve all 10 seeds (plain REINFORCE
misses one), and it does so no later — mean solve episode ≈229 for both, over the seeds
plain REINFORCE solves at all. The gain is not in how fast it gets there; it is in what
happens after.

That is what a real baseline buys. V(s) is a state-dependent estimate of expected return,
so G_t − V(s_t) is a genuine advantage — positive exactly for actions that did better than
the state warranted — rather than the positional artefact of §2. Same gradient in
expectation, much less variance in the sample.

Its one weak seed fails differently from the other methods' weak seeds: it never climbed
past a 335 peak, rather than climbing to 500 and then collapsing.

### 2. The scalar baseline does not help — and hurts stability

Worst of the three on final score, and it collapses from its peak far more often (only
4/10 seeds hold; one goes 492 → 82). It is also the slowest to solve (mean episode 300 vs
≈229). An earlier single-pair run suggested it roughly halved training time; that was a
two-seed fluke plus a bug in the stopping condition, and the 10-seed picture does not
support it.

The reason is structural. With one episode per update and gamma = 1.0, the return at
timestep t is just the number of steps remaining, so subtracting the episode mean gives

    advantage(t) = (T - t) - (T + 1) / 2

which depends only on position t within the episode, not the state or action. It is a
fixed positional ramp — it removes the constant positive offset but injects a new variance
source, because it is recomputed each episode and its scale shifts as episode length
changes between updates. With no trust region, that makes destructive updates more likely.
The value baseline fixes exactly this: V(s) varies with the state and is trained toward a
stable target, rather than being a per-episode constant.

### 3. Catastrophic forgetting is reduced, not removed

Every seed of every method touches 500 at some point, and for two of the three methods
several then fall back for good: plain REINFORCE ends seeds at 500 → 376 and 461 → 307,
the scalar baseline at 492 → 82 and 492 → 239. This is the defining failure of vanilla
policy gradient: nothing constrains how far the policy moves in one update, so one unlucky
batch can step off a cliff and not recover within the run.

The value baseline largely removes forgetting at that scale — 9/10 seeds hold ≥90% of
their peak and the mean drop is halved. It does not remove the underlying problem, which
is still plainly visible per episode: even on seeds that end around 480, individual
episodes after the first 500-step run still fall to 15–30 steps, and the weakest seed
spends 56% of its post-500 episodes below 200. A lower-variance gradient makes the
destructive step rarer without bounding it.

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

For the value baseline, two details are load-bearing:

- **V(s) is detached in the policy loss** (`advantages = returns - values.detach()`). The
  baseline theorem holds only if the baseline is a constant with respect to the policy
  gradient; V trains separately through an MSE loss toward the observed returns.
- **The value network's parameters must actually be in the optimizer**
  (`Adam(list(policy.parameters()) + list(value_net.parameters()), lr=lr)`). An earlier
  version passed only the policy's parameters, so the gradient computed for V was never
  applied and the "learned" baseline stayed frozen at its random initialisation for all
  600 episodes. Nothing errors, and the run still beat plain REINFORCE slightly — a fixed
  random state-dependent function is still a valid, if useless, baseline. The numbers
  above are from the fixed version; the fix turned a marginal win into a win on every
  column.

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
