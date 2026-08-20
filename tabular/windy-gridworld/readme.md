## Windy Gridworld (Exercises 6.9 & 6.10)

### Problem
The Windy Gridworld is a standard gridworld with a crosswind running upward  
through the middle of the grid. The agent must navigate from a start state  
to a goal state while being pushed upward by wind of varying strength.

### Implementations

**Exercise 6.9 — Standard Windy Gridworld (4 actions)**
- Algorithm: On-policy SARSA (TD control)
- Actions: up, down, left, right
- Grid: 7×10 with wind strengths [0,0,0,1,1,1,2,2,1,0]
- Result: Optimal path found in ~15 steps

**Exercise 6.10 — Extensions**
- King moves (9 actions): Added diagonal moves + no-action  
  → Optimal path reduced to **7 steps**
- Stochastic wind: Wind strength varies by ±1 randomly each step  
  → Agent adapts and finds goal in ~16 steps

### Key Results
- Standard SARSA converges to the book's optimal 15-step path
- King moves significantly improve performance (7 vs 15 steps)
- Stochastic wind increases path length but agent still converges

### Dependencies
- Python 3.x
- NumPy
- Matplotlib

---

## Reference
Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction*  
(2nd ed.). MIT Press. http://incompleteideas.net/book/the-book-2nd.html
