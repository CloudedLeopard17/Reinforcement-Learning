"""
Maximization bias, and how far it propagates.

Setup: an episode of L steps where the action NEVER matters. At the final
step the agent receives +1 or -1 with equal probability. Every state-action
pair therefore has a true value of exactly 0 -- a correct Q-function says 0
everywhere, and there is nothing to learn.

Tabular Q-learning still drifts upward, because max over noisy estimates is
biased high: E[max of estimates] > max of true values. Each Bellman backup
injects a small positive bias, and the backup carries that bias to the
previous step multiplied by gamma.

Run it to see when gamma matters and when it doesn't.
"""

import numpy as np

N_ACTIONS = 5
ALPHA = 0.05


def run(gamma, L, episodes, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((L, N_ACTIONS))          # true value is 0 for every entry

    for _ in range(episodes):
        for s in range(L):
            a = rng.integers(N_ACTIONS)   # action is irrelevant by construction
            if s == L - 1:
                target = 1.0 if rng.random() < 0.5 else -1.0   # coin flip
            else:
                target = 0.0 + gamma * Q[s + 1].max()          # bootstrap
            Q[s, a] += ALPHA * (target - Q[s, a])

    return Q[0].max()                     # should be 0; it won't be


if __name__ == "__main__":
    print("Learned Q at the start state (true value = 0 in every row)\n")
    print(f"{'episode len':>12} {'gamma=0.9':>11} {'gamma=0.99':>12} {'ratio':>8}")
    for L, episodes in [(2, 200_000), (5, 100_000), (20, 40_000), (50, 20_000)]:
        lo = run(0.9, L, episodes)
        hi = run(0.99, L, episodes)
        print(f"{L:>12} {lo:>11.3f} {hi:>12.3f} {hi / lo:>8.1f}")

    print("""
Reading the table:

- Both discounts overestimate. That is maximization bias, and it exists
  regardless of gamma.
- With 2-step episodes the two discounts are almost identical. The bias only
  gets multiplied by gamma once, so 0.9 vs 0.99 barely differ.
- The gap opens up as episodes get longer, because the bias compounds once
  per backup. Over 50 steps, 0.9 has decayed it to nothing while 0.99 has
  barely touched it.

The 1/(1-gamma) amplification is a long-horizon effect. It needs an episode
long enough for the geometric series to actually run.
""")
