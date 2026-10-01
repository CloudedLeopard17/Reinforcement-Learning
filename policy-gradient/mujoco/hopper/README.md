# DDPG vs TD3 on Hopper

[Pendulum](../../pendulum/README.md) showed DDPG converging tightly on all ten seeds — no collapse,
no overestimation, nothing for TD3 to fix. Hopper is the environment the TD3 paper makes its case
on, and the same implementation behaves completely differently once the state goes from 3 dims to
11 and the action from 1 to 3.

Both arms are the same code with `use_td3` toggled, so exactly three things differ: **twin critics**
taking the elementwise minimum in the target, **target policy smoothing**, and **delayed** actor and
target updates. Same 5 seeds, same OU exploration, same hyperparameters, same fixed evaluation
starts, 1M environment steps each (~10 h per arm with the seeds running in parallel).

## Result

| | DDPG | TD3 |
|---|---|---|
| final greedy return | 1935 (sd 978) | **3454** (sd 157) |
| paired difference | — | **+1518 ± 839** (95% CI, n=5) |
| swing between adjacent checkpoints | 548 | **206** |
| 2nd-half checkpoints below 1500 | 64.8% | **2.0%** |
| critic ÷ true discounted return | 1.25 – 1.55× | **0.96 – 1.03×** |

Both arms land on the published numbers — the TD3 paper reports roughly 1860 for DDPG and 3560 for
TD3 on Hopper at 1M steps. Neither clears Gymnasium's `reward_threshold` of 3800. For scale,
`healthy_reward` is 1.0 per step against a 1000-step cap, so **~1000 means "learned not to fall"
without moving** — a local optimum, not half a solution.

### 1. TD3 doubles the score and collapses the seed variance

| seed | DDPG | TD3 | Δ |
|---|---|---|---|
| 0 | 1696 | 3631 | +1935 |
| 1 | 978 | 3224 | +2245 |
| 2 | 1332 | 3447 | +2116 |
| 3 | 2177 | 3562 | +1385 |
| 4 | 3493 | 3403 | −90 |

Four of five seeds improve by 1385 to 2245 points, and the spread falls from sd 978 to sd 157.

Seed 4 is the honest footnote: it was DDPG's one good run, and TD3 does not beat it. **The gain is
not a higher ceiling — it is making every seed behave like the lucky one.** DDPG could already
reach 3400; it just couldn't stay there or do it reliably.

### 2. The instability is gone — but mind which number says so

DDPG's greedy score swung 548 points between adjacent checkpoints and spent 64.8% of second-half
checkpoints below 1500; seed 2 fell from 3333 to 361, a competent hopper reduced to falling over.
TD3 swings 206 and spends 2.0% below 1500. All five TD3 seeds exceed 3000 and none ends below 2500.

One trap worth recording. The *ratio* of the swing to each arm's own evaluation noise is 3.6× for
DDPG and **8.9× for TD3** — which reads as TD3 being less stable, and isn't. TD3's denominator
shrank: its policy scores consistently across all ten fixed starts, so the within-checkpoint
standard error fell from 150.6 to 23.1. That ratio answers "is this arm's movement real rather than
evaluation noise", *within* an arm. Across arms, compare the absolute swing. Same family of mistake
as the peak-vs-final bias caught on [Pendulum](../../pendulum/README.md) — a statistic that is
correct for one question and misleading for a neighbouring one.

### 3. The twin critics do exactly what they claim

Both arms measured identically: load the saved actor, roll it out on the ten fixed start states,
discount at γ=0.99, and compare against the critic's own logged estimate.

| seed | DDPG ratio | TD3 ratio |
|---|---|---|
| 0 | 1.35× | 1.00× |
| 1 | 1.30× | 0.99× |
| 2 | 1.29× | 1.03× |
| 3 | **1.55×** | 0.99× |
| 4 | 1.25× | 0.96× |
| mean | 1.35× | **0.99×** |

DDPG's critic was 25–55% above the return its own policy achieved. TD3's is calibrated to within
4%. This is the mechanism the paper asserts, measured rather than assumed — and it is the first
time this repo has caught a critic that was *merely optimistic* rather than provably broken.
Pendulum's `Q ≤ 0` ceiling could only do the latter, and on Hopper no such bound exists, so the
rollout comparison is the only way to see it.

Reading it alongside DDPG's logged Q makes the failure concrete: DDPG's critic reported nearly the
same value (316–347) for policies whose actual performance differed 3.5-fold. It had stopped
tracking the policy at all.

## What changed from the Pendulum code

Mostly shapes, but four things had to change and each would have failed quietly:

- **Hopper's observations are `float64`**, Pendulum's were `float32`, so `torch.tensor(state)` makes
  a double tensor and the first forward pass dies. Loud, at least — unlike the rest.
- **`max_action` must be 1.0, not 2.0.** Keeping Pendulum's `2 * tanh(...)` lets the actor propose
  actions outside the legal range, which `np.clip` truncates before the env sees them — so the
  critic learns from clipped actions while the actor is optimised at points the critic never saw.
- **The OU noise needs one dimension per joint.** A scalar process broadcasts across the action
  vector and perturbs all three joints identically: one shared exploration direction.
- **The loop counts steps, not episodes.** Episodes range from ~20 steps (falling) to 1000, so an
  episode budget would silently expand as the agent improved.

Writing the TD3 arm turned up four more of the same kind, all of which run without erroring:
a `target` assigned only on the DDPG branch; the second critic's parameters left out of the
optimizer (the [REINFORCE value-baseline bug](../../cartpole/README.md) again); both critics built
with the same seed, so `min(Q1, Q2)` was identically `Q1` and TD3 silently reduced to DDPG; and
target smoothing applied to the exploration action instead of the target action.

Two more that are properties of the setup rather than porting errors: `terminated` is genuinely
terminal here (the hopper falls) while the 1000-step cap arrives as `truncated` and must still
bootstrap — the first environment in the repo where both really happen; and episode counts differ
across seeds (4335 to 6300), so per-episode arrays cannot be stacked and each seed writes its own
file.

## Running it

```bash
pip install "gymnasium[mujoco]" torch numpy matplotlib pandas
jupyter notebook hopper.ipynb
```

Set `USE_TD3` and run the sweep cell; each arm writes `results/{ddpg,td3}_seed{n}.npz` plus actor
and critic checkpoints. The five seeds run as five forked processes, each with its own env and 4
torch threads. **CPU is deliberate**: measured at 164 steps/s against 145 on CUDA, because a 256×256
network at batch 256 cannot fill a GPU. It also avoids the "cannot re-initialise CUDA in a forked
subprocess" failure.

Wall clock was ~10 h per arm, against ~3.5 h estimated. The difference is the replay buffer:
`random.sample` on a `deque` indexes into the middle, which is O(n), so cost grows as it fills —
0.18 ms per batch at 30k entries, **14.1 ms at 1M**. Pre-allocated NumPy arrays sample the same
batch in 0.49 ms. The deque was kept for TD3 so both arms share identical buffer code; replacing it
is the obvious speedup for anything after this.

The `.npz` files are committed — they are the data behind every number above. The `.pth`
checkpoints are gitignored; the notebook regenerates them.
