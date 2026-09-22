# DDPG on Pendulum: the first continuous-action rung

Every other policy in this repo outputs a `Categorical`. DDPG outputs a number: a deterministic
actor maps Pendulum's 3-dim state to a torque in [−2, 2], and a critic Q(s, a) supplies the
gradient that tells the actor which way to move it. Most of the machinery is DQN's — replay
buffer, target networks, Polyak averaging — so what's genuinely new is small: a `tanh`-bounded
actor, a critic that takes the action as an input, and exploration by adding noise rather than
sampling.

Compared over **10 seeds**, which Pendulum's minutes-per-run makes affordable again after
[Pong](../pong/README.md) forced n = 1.

## Result

| | |
|---|---|
| final greedy return, mean over seeds | **−113.9** |
| per-seed range | −110.5 to −117.3 |
| change, episodes 109–149 → 159–199 | **−0.7 ± 1.0** — no collapse |
| change, episodes 59–99 → 159–199 | +30.6 ± 11.6 — still improving to ~episode 100 |
| max mean Q, any seed or episode | −5.69, against a hard ceiling of 0 |

Greedy evaluation: no noise, separate env instance, 10 fixed start states, every 10 episodes.

All ten seeds converge, tightly, and stay converged. The critic never breaks the Q ≤ 0 ceiling.
DDPG's reputation for reaching a good score and then falling apart doesn't show up here — and the
two things that looked like it earlier were both measurement artefacts, which is the more useful
part of this write-up.

### 1. Peak-vs-final manufactures a collapse

An earlier version compared each seed's *maximum* 20-episode rolling mean with its final 20
episodes, and reported every seed "reaching" about −124 and "ending" at −176: a 52-point drop,
exactly what a DDPG collapse should look like.

It isn't one. The maximum of ~180 overlapping noisy windows sits above the series mean even when
nothing is changing. Bootstrapping a stationary run from the converged training returns — zero
forgetting by construction — the same metric still reports an expected drop of **43.6** (90%
range 5–92). The observed 52 is inside that.

The metric worked in the [REINFORCE comparison](../cartpole/README.md) because those drops were
400+ points against the noise (500 → 82). Pendulum's per-episode sd of ~117 swallows a 50-point
effect. Comparing two *fixed* windows has no selection step, and gives the flat answer above.

### 2. Fixed evaluation seeds shrank the interval nineteenfold

The window comparison on the earlier version, with random evaluation starts, gave **−1.8 ± 19.2**.
The current one gives **−0.7 ± 1.0**. Same algorithm, same ten seeds — only the pairing changed.

Pendulum resets the angle uniformly in [−π, π], and the start state dominates the return. Final
checkpoint, per evaluation start, averaged over seeds:

```
[-3, -123, -130, -125, -124, -119, -251, -127, -133, -3]
```

Two starts are essentially upright, one is a hard swing-up, the rest cost about −125. With random
starts every checkpoint is a fresh draw from that spread; with fixed starts, episode *i* is the
same situation at every checkpoint and the comparison is paired. Keeping all 10 per-episode returns
rather than their mean is what made the structure visible.

### 3. What this means for TD3

There's nothing here for it to fix: no collapse, no overestimation past the ceiling, tight
convergence. A TD3 comparison on Pendulum would most likely be null. The TD3 paper makes its case
on MuJoCo tasks (HalfCheetah, Hopper, Walker) where DDPG's critic does overestimate and its
performance degrades — that's where the comparison belongs.

## Bugs that broke a working version, silently

Each of these produced a notebook that ran to completion, printed plausible numbers, and was wrong.

**The actor loss must pass through the actor.** `-critic(states, actor(states))`, not
`-critic(states, actions)` with the buffer's actions. With buffer actions there's no path from the
actor's parameters to the loss, so `actor.grad` stays `None` and Adam silently skips the step. It
doesn't error, because the loss still requires grad through the critic's own weights. The actor sat
at its random initialisation for a whole run (average return −1537).

**The actor trains on the replay batch.** An earlier version computed the actor loss on the single
state just taken from the environment — a batch of one, from consecutive correlated states, while
the critic trained on 64.

**Store `done`, never `done or truncated`.** Pendulum never terminates — over 2000 steps it reports
`terminated=0, truncated=10` — so storing the time limit as terminal turned *every* episode boundary
into a zero-value target. Same rule as `done = not interrupted` in the
[Unity notebooks](../../unity/README.md) and `terminated` vs `terminated | truncated` in the
[Pong one](../pong/README.md); Pendulum is where it bites hardest, since 100% of episode ends are
truncations.

**The actor's backward fills the critic's gradients.** The critic is in the graph, so its `.grad`
buffers receive the actor's gradient. Only the actor's optimizer steps, so nothing is applied — but
the gradients sit there, and `optimizer_critic.zero_grad()` has to run before the critic's own
backward.

**OU noise at `dt=0.01` isn't exploration.** Its correlation time is `1/(θ·dt)` = 667 steps against
a 200-step episode, so within an episode it's a slowly drifting constant offset (measured at
−0.57 … +0.09 across one episode), and without a per-episode `reset()` that offset carries over.
`dt=0.05` and a reset per episode fix it; plain Gaussian noise is a fine alternative.

Two logging bugs as well, both of the "right shape, wrong data" kind: per-episode lists returned
where per-run ones were meant (the Q-value plot showed the final episode's 200 steps under an
"Episode #" axis — undetectable by shape, since the episode length and `n_episodes` are both 200),
and a per-run list initialised inside the episode loop, so only the last checkpoint survived.

## The critic's bound

Reward is `−(θ² + 0.1θ̇² + 0.001a²)`, which lies in [−16.27, 0], and episodes are exactly 200 steps.
With γ = 0.99:

```
Q ∈ [−1409, 0]
```

The upper end is the useful one: any positive Q is provably wrong, no arithmetic needed. Mean Q
per episode rises to about −24 early, settles around −45 to −52, and never goes above −5.69.

## Configuration

| | |
|---|---|
| actor | 3 → 256 → 256 → 1, `2·tanh` output |
| critic | state → 256, concat action → 256 → 256 → 1 |
| episodes × steps | 200 × 200 = 40k environment steps per seed |
| γ / τ | 0.99 / 1e-3, Polyak every step |
| learning rate | actor 3e-4, critic 5e-4 |
| replay | 50k transitions, batch 64, updates from step 500 |
| exploration | OU noise, σ 0.2, θ 0.15, dt 0.05, reset per episode |

## Running it

```bash
pip install "gymnasium[classic-control]" torch numpy matplotlib pandas
jupyter notebook ddpg.ipynb
```

CPU is fine. Results for all ten seeds are saved to `ddpg_pendulum.npz` — per-episode scores, Q
values and losses, and the full `(seeds, checkpoints, eval episodes)` greedy-evaluation array.
