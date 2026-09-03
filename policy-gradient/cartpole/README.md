# Policy gradient on CartPole: from REINFORCE to PPO

Two notebooks on one environment, one change per rung, everything written from scratch in
PyTorch and compared over **10 seeds**. The seed count is not decoration: CartPole is
seed-sensitive enough that an earlier two-run comparison gave the *opposite* conclusion to
the one below.

| Notebook | What it adds | Headline |
|---|---|---|
| [`reinforce.ipynb`](reinforce.ipynb) | no baseline / scalar baseline / learned value baseline | the value baseline wins on every column — but all three still collapse after reaching 500 |
| [`ppo_rollout.ipynb`](ppo_rollout.ipynb) | clipped surrogate objective + fixed-length rollouts | 7/10 seeds end greedy-perfect (500.00, zero variance) and nothing reaches 500 then crashes |

---

# Part 1 — REINFORCE: does a baseline help?

Three variants of Monte Carlo policy gradient:

1. **Plain REINFORCE** — advantage is the raw return G_t.
2. **Scalar baseline** — subtract the mean return of the current episode, G_t − mean(G).
3. **Value baseline** — subtract a learned V(s_t), G_t − V(s_t). This is the critic.

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

---

# Part 2 — PPO: bounding the update

Two changes from the value-baseline REINFORCE, both aimed at the collapse above:

- **Clipped surrogate objective.** The ratio π_new/π_old is clipped to [1−ε, 1+ε] and the
  objective takes the *minimum* of the clipped and unclipped terms —
  `-min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)`. Taking the min of the two *objectives* rather
  than the two ratios is what makes it behave for both signs of the advantage: it caps how
  far a good action is pushed up and a bad one pushed down, while leaving unfavourable
  moves unclipped so they can still be corrected.
- **Fixed-length rollouts (2048 steps).** One episode per update means the batch size is a
  function of policy skill — it grows from ~20 steps to 500 as the agent improves. A fixed
  rollout decouples the two and gives each update a diverse batch spanning several
  episodes instead of one correlated trajectory.

Clipping is also what makes the inner loop safe: each rollout is reused for 4 epochs, which
is where PPO's sample efficiency comes from, and that is only sound because the policy
cannot wander far from the data that produced it.

## Result

The training curve is the mean return of the *stochastic* policy during collection, and the
0.01 entropy bonus keeps that policy deliberately non-deterministic. Greedy evaluation
(argmax actions, 30 episodes per seed, after training) is the honest measure:

| seed | greedy mean ± std | training score (last 30 updates) |
|---|---|---|
| 0, 1, 2, 3, 5, 7, 8 | **500.00 ± 0.00** | 156 – 439 |
| 4 | 494.70 ± 11.65 | 365 |
| 6 | 413.30 ± 79.74 | 211 |
| 9 | 200.87 ± 30.72 | 417 |

**7 of 10 seeds end at a perfect 500.00 with zero variance**, an eighth within 5 points of
it. Overall greedy mean across all 300 evaluation episodes: 461.

**The single clearest result is seed 5.** Its training score ends at 156 — which looks
exactly like the catastrophic forgetting of Part 1. Its greedy score is a flawless 500.00
± 0.00. The collapse was entirely exploration noise from the entropy bonus; the policy
underneath was intact the whole time. Training curves lie about a stochastic policy in a
way greedy evaluation does not. Part 1's stability numbers are training scores too, so they
are pessimistic in the same direction — though less so, since those runs have no entropy
bonus holding the policy stochastic. (Both notebooks' *solve* checks are already greedy.)

### The failure mode changed

This is the actual finding, and it is not about the score. PPO's two failures are seeds
that never escaped a mediocre policy — seed 9 sits at a *stable* 200.87 ± 30.72,
consistently stuck rather than falling apart. Not one seed shows the reach-500-then-crash
pattern that every REINFORCE variant showed. The clip bounds exactly the step that causes
it, so a policy that gets to 500 stays there.

CartPole is too easy to separate these methods on raw score — everything lands at median
500 with a few unlucky seeds. Stability is the axis that separates them, and on that axis
PPO does precisely what it was designed to do.

### It is not cheaper here

PPO takes roughly **131k environment steps** to hit the solve criterion, against ~30k for
the REINFORCE variants — about 4× more interaction (approximate: the check runs every 10
updates, so the granularity is 20k steps). Fixed 2048-step rollouts collect far more data
per update than a single short episode, and on a task this easy that data is mostly
redundant. What the extra interaction buys is the stability above, not speed.

### Notes and caveats

- **`dones` records `terminated`, not `truncated`.** An episode cut at the 500-step cap is
  not a terminal state — the pole was still up — so its return must bootstrap through the
  cut. Zeroing it teaches the value network that surviving to 500 has no future value.
- **The rollout edge bootstraps too.** `R = next_value` seeds the backward recursion, so
  the unfinished episode at the end of each rollout is not treated as a zero-value
  terminal. `R = 0` scored marginally higher on some seeds, but that is a within-noise
  accident of a biased target — and it would be badly wrong on Atari, where nearly every
  rollout cuts mid-episode.
- **The state carries across rollouts**, so consecutive rollouts continue the same episode
  stream and no transitions are wasted at the boundary.
- **Advantage is Monte Carlo (returns − V), not GAE.** GAE is wired in (`lam`, the
  bootstrap value) but unused; it did not improve CartPole. It is the right estimator once
  rollouts are long and never align with episode boundaries.
- **lr = 0.005 is high for PPO** (the usual default is ~3e-4). CartPole tolerates it with
  gradient clipping to norm 1.0; it will not transfer to Atari.
- **One rollout per update, single environment.** Atari needs vectorised environments for
  on-policy throughput — the same rollout logic across several envs in parallel.

---

## Methodology

Everything is held fixed between the three REINFORCE variants except the baseline: the same
seed list (policy init, NumPy, environment RNG), the same `.sum()` reduction and learning
rate, and the same greedy-evaluation stopping rule. PPO shares the seed list, the network
sizes, the learning rate, and the stopping rule, but changes gamma (0.99 vs 1.0) along with
the batching, so it is a rung on the ladder rather than a controlled A/B against Part 1.

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
  gradient; V trains separately through an MSE loss toward the observed returns. The same
  applies to the old log-probs and advantages in the PPO ratio — gradient must flow only
  through the current policy.
- **The value network's parameters must actually be in the optimizer**
  (`Adam(list(policy.parameters()) + list(value_net.parameters()), lr=lr)`). An earlier
  version passed only the policy's parameters, so the gradient computed for V was never
  applied and the "learned" baseline stayed frozen at its random initialisation for all
  600 episodes. Nothing errors, and the run still beat plain REINFORCE slightly — a fixed
  random state-dependent function is still a valid, if useless, baseline. The numbers
  above are from the fixed version; the fix turned a marginal win into a win on every
  column.

## Hyperparameters

| | REINFORCE variants | PPO |
|---|---|---|
| policy / value net | 32-unit hidden / 16-unit hidden | same |
| gamma | 1.0 | 0.99 |
| learning rate | 0.005 | 0.005 |
| batch | 1 episode | 2048-step rollout |
| epochs per batch | 1 | 4 |
| clip ε | — | 0.2 |
| entropy / value coefficients | — / 0.5 | 0.01 / 0.5 |
| gradient clipping | — | norm 1.0 |
| budget per seed | 600 episodes | 400 updates (819k steps) |

## What comes next

**A2C** is the rung this pair skips: bootstrap the return instead of waiting for the full
episode. After that, the same PPO rollout logic scales to Atari — vectorised environments,
GAE, and a learning rate an order of magnitude lower.

## Running it

```bash
pip install "gymnasium[classic-control]" torch numpy matplotlib pandas
jupyter notebook reinforce.ipynb    # Part 1
jupyter notebook ppo_rollout.ipynb  # Part 2
```

CPU is fine — CartPole episodes are short and the networks are tiny.
