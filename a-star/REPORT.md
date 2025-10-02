# Report: A* on the Sudoku Cube

## Program overview
This project implements A* search to solve a Sudoku Cube, plus a k-randomized experiment suite and plotting tool.

### What runs
- `a-star.py` – all code.
- Interactive CLI menu drives all tasks.

---

## Data structures used in the **search algorithm**
- **State:** `Cube` object with six `Face` objects (`top, bottom, left, right, front, back`).
- **Move:** `(face, direction)` where `direction` is `{clockwise, counterclockwise}`.
- **Neighbors:** `Cube.neighbors(skip_inverse_of=...)` yields `((face, dir), next_cube)` for the 12 possible quarter-turns; optionally invalidates the immediate inverse of the previous move.
- **Goal test:** `Cube.is_solved()` — each face contains digits `1-9` exactly once.
- **State key:** `Cube.state_key()` → tuple of 54 ints in fixed face order (TOP, LEFT, FRONT, RIGHT, BACK, BOTTOM) for hashing/checking what's been visited.
- **Node:** `(cube, g, move, parent)` with `.path()` reconstruction.
- **Frontier:** heap using `(f = g + h, g, counter, node)`.

---

## Heuristic used (and why it's admissible)
We use `Cube.heuristic` (already in code) which computes:

1. For each face, the deficit = `9 − (# of distinct digits on that face)`.
2. Lower bound A: `ceil(max_face_deficit / 3)`  
   - In one quarter-turn, any affected adjacent face has at most one row or one column (3 tiles) replaced. Therefore, a single move can introduce 3 or less new correct digits to any single face. Fixing the most deficient face thus needs at least `deficit/3` moves.
3. Lower bound B: `ceil(sum_of_all_face_deficits / 12)`  
   - A quarter-turn moves four strips of length 3 around the "ring" (12 tiles total). So, globally, a single move can fix 12 or less missing/duplicate digits across the whole entire cube.
4. Heuristic value `h(s) = max(A, B)` - the max of valid lower bounds is itself also a valid lower bound.

Because each bound counts a max number of tiles that could be corrected by 1 move, neither bound can overestimate the true distance. Therefore `h` is admissible (`h(s) less than or equal to optimal_cost(s)` for all `s`).

---

## Randomizer
`apply_random_moves(cube, k, seed, only_cw=True, prevent_undo=True)` applies `k` random CW turns (no immediate inverse). We **do not** expose the generation path to the solver; A* must rediscover a solution.

---

## A* implementation details
- **Cost model:** each quarter-turn costs 1 (uniform).
- **Expansion:** we optionally prune the immediate inverse of the last move (safe).
- **Direction filter at solve time:** `allowed_dirs=(CCW,)` so A* expands only CCW moves, matching the instructor hint. This preserves optimality for CW-only scrambles and ensures the optimal cost equals `k`.

**Instrumentation (for the required metric):**
- We count expansions per `f=g+h` value. Upon reaching the goal, `f* = g* = k` (since `h(goal)=0`).  
  The “nodes expanded in the **last iteration** of A*” equals the count at `f*`.

---

## How to compile/run
- See **README** for environment setup and the interactive menu.
- Typical flow:
  1. Run experiments (`k_max`, trials, base seed) -> CSV.
  2. Analyze & plot -> summary CSV and PNG.

---

## Results
- **Figure:** `astar_plot_20251001_235642.png`  
- **Trend:** average “last-iteration” nodes grows rapidly with `k` (see the plot).  
- **Table:**
Summary (avg nodes expanded in LAST iteration):
 k | trials | solved | avg_last_iter
---+--------+--------+--------------
 3 |      5 |      5 |        10.40
 4 |      5 |      5 |        41.20
 5 |      5 |      5 |       211.00
 6 |      5 |      5 |      1072.80
 7 |      5 |      5 |      5831.00
 8 |      5 |      5 |     31582.00
---

## What I learned
- Implementing A* requires a clear **state key** to dedupe and a correct **goal test**.
- An admissible heuristic can be engineered from **move-locality bounds** (≤3 per face, ≤12 global).
- Even with an admissible heuristic, branching grows quickly; restricting directions (CW scramble, CCW solve) keeps the metric tight and experiments feasible.

---

## Collaboration / sources
- Partner(s): [listed in Canvas comments]
- Borrowed GUI/util code: Matplotlib used for plotting
- LLM usage: We used an AI assistant (ChatGPT) to explain A* and draft some text for this report. All core modeling/solver/randomizer code is mine/ours."
