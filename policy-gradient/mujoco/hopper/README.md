# DDPG on Hopper: where the baseline actually breaks

[Pendulum](../../pendulum/README.md) showed DDPG converging tightly on all ten seeds — no
collapse, no overestimation, nothing for TD3 to fix. Hopper is the environment the TD3 paper makes
its case on, and the same implementation behaves completely differently once the state goes from 3
dims to 11 and the action from 1 to 3.

5 seeds × 1M environment steps, ~10 hours wall clock with the seeds running in parallel.

## Result

| | |
|---|---|
| mean greedy return at 1M | **1935** |
| per seed | 1696, 978, 1332, 2177, 3493 (sd 978) |
| best checkpoint per seed | 3438, 1799, 3333, 3227, 3493 |
| seeds that exceeded 3000 | 4/5 — of which 3 ended below 2500 |
| critic overestimation | **1.25× – 1.55×** its own policy's discounted return |

Gymnasium's `reward_threshold` for Hopper-v5 is **3800**, so nothing here is solved. `healthy_reward`
is 1.0 per step against a 1000-step cap, which means **~1000 is "learned not to fall" without
moving** — a local optimum rather than half a solution. The TD3 paper reports roughly 1860 for DDPG
at this budget, so 1935 lands where it should.

### 1. The instability is real, not evaluation noise

This is the claim [Pendulum](../../pendulum/README.md) taught us to check before making. There, an
apparent 52-point collapse turned out to be selection bias in the metric. Here the effect clears
the noise floor by a wide margin:

```
SE of a single checkpoint (10 eval episodes) :  150.6
mean |change| between adjacent checkpoints   :  548.3   (3.6x the noise floor)
```

Per seed, over the second half of training:

| seed | best | at | final | 2nd-half range | checkpoints below 1500 |
|---|---|---|---|---|---|
| 0 | 3438 | 860k | 1696 | 25 – 3438 | 68% |
| 1 | 1799 | 650k | 978 | 703 – 1799 | 86% |
| 2 | 3333 | 550k | 1332 | 361 – 3333 | 76% |
| 3 | 3227 | 380k | 2177 | 341 – 2739 | 64% |
| 4 | 3493 | 1000k | 3493 | 318 – 3493 | 30% |

Seed 2 goes from 3333 to 361: a competent hopper reduced to falling over. Only seed 4 ends at its
own best. This is the catastrophic forgetting the [CartPole PPO notebook](../../cartpole/README.md)
attributed to unbounded policy updates, now appearing in an off-policy actor-critic.

### 2. The critic overestimates, and it is measurable

Logged mean Q at the end of training is nearly identical across seeds while actual performance
differs 3.5-fold — the critic predicts roughly the same value whether its policy scores 978 or
3493. Rolling out the saved actors on the fixed evaluation starts and discounting properly:

| seed | undiscounted | true discounted | mean Q | ratio |
|---|---|---|---|---|
| 0 | 1829 | 234 | 316 | 1.35× |
| 1 | 982 | 254 | 330 | 1.30× |
| 2 | 1255 | 263 | 339 | 1.29× |
| 3 | 2502 | 221 | 343 | **1.55×** |
| 4 | 3272 | 278 | 347 | 1.25× |

This is the bias TD3's twin critics target, and it is the first time in this repo it has been
*measured* rather than bounded. Pendulum's provable `Q ≤ 0` ceiling could only catch a critic that
was already broken; it had nothing to say about one that was merely optimistic.

Caveat, stated because it affects how much weight the table carries: the logged Q averages over
replay states drawn from a mixture of past policies, while the discounted return starts from the
10 fixed eval states under the final policy. Saving the critic alongside the actor would let both
sides be computed on identical states — worth doing for the TD3 run.

## What changed from the Pendulum code

Mostly shapes, but four things had to change and each would have failed quietly:

- **Hopper's observations are `float64`**, Pendulum's were `float32`. `torch.tensor(state)` then
  produces a double tensor and the first forward pass dies with `mat1 and mat2 must have the same
  dtype`. Loud, at least — unlike the rest.
- **`max_action` must be 1.0, not 2.0.** Keeping Pendulum's `2 * tanh(...)` lets the actor propose
  actions outside the legal range, which `np.clip` truncates before they reach the env — so the
  critic learns from clipped actions while the actor is optimised at points the critic never saw.
- **The OU noise needs one dimension per joint.** A scalar process broadcasts across the action
  vector and perturbs all three joints identically: one shared exploration direction.
- **The loop counts steps, not episodes.** Episodes range from ~20 steps (falling) to 1000, so an
  episode budget would silently expand as the agent improved.

Two more that are properties of this setup rather than porting errors: `terminated` is genuinely
terminal here (the hopper falls) while the 1000-step cap arrives as `truncated` and must still
bootstrap — the first environment in the repo where both really happen; and episode counts differ
across seeds (5064 to 6300), so per-episode arrays cannot be stacked and each seed writes its own
file.

## Running it

```bash
pip install "gymnasium[mujoco]" torch numpy matplotlib pandas
jupyter notebook hopper.ipynb
```

The five seeds run as five forked processes, each with its own env and 4 torch threads. **CPU is
deliberate**: measured at 164 steps/s against 145 on CUDA, because a 256×256 network at batch 256
cannot fill a GPU. It also avoids the "cannot re-initialise CUDA in a forked subprocess" failure.

Wall clock was ~10 h for all five in parallel, against ~3.5 h estimated. The difference is the
replay buffer: `random.sample` on a `deque` indexes into the middle, which is O(n), so cost grows
as it fills — 0.18 ms per batch at 30k entries, **14.1 ms at 1M**. Pre-allocated NumPy arrays
sample the same batch in 0.49 ms. Worth changing before the TD3 run.

Results land in `results/` as one `.npz` plus one `.pth` per seed. The `.npz` files (~500 KB total)
are committed, as `reinforce_comparison.npz` and `ddpg_pendulum.npz` are — they are the data behind
every number above. The `.pth` checkpoints are gitignored; the notebook regenerates them.

## Next

TD3: twin critics with the minimum, delayed actor updates, and target policy smoothing — three
changes to this code, everything else held fixed, same 5 seeds and same evaluation starts.
