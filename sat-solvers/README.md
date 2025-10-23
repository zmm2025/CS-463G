# Algorithm Descriptions & Design Decisions
- **GSAT** (incomplete/heuristic): random initial assignment, greedy variable flips to maximize satisfied clauses; `max_flips` set to 2500 to reduce runtime. Note: not guaranteed to find satisfying assignment even if one exists.
- **WalkSAT**  (incomplete/heuristic): random initial assignment; at each step choose an unsatisfied clause, then flip a variable at random with probability `p` (random walk), else flip the variable giving best improvement; `max_flips` set to 2500 to reduce runtime, `p` set to 0.5.
- **DPLL** (complete): unit propogation, pure literal elimination (if peresent), backtracking, 10-second timeout per instance. Status is SAT if all clauses satisfied, UNKNOWN on timeout; UNSAT only of proof reached.
- **Other implementation notes**
  - Used `time.perf_counter()` for CPU timing across all algorithms
  - Added guards to skip MacOS and metadata files when loading benchmarks
  - Filtered out bad/empty NCFs to prevent parsing errors
  - Added extra safeguards in WalkSAT to safely skip empty clauses
  - Used deterministic seeding (`--seed`) for reproducible heurstic testing

# Code & Data
- `sat_runner.py` - all 3 algorithms: GSAT, WalkSAT, and DPLL
- Data
  - `gsat-2500.csv` - GSAT run on uf20, uf50, and uf75 benchmarks using `max_flips=2500`
  - `walksat-2500.csv` - WalkSAT run on uf20, uf50, and uf75 benchmarks using `max_flips=2500`
  - `dpll-10.csv` - DPLL run on uf20, uf50, and uf75 benchmarks using `timeout=10.0`
- Graphs
  - `heuristics_overall_runtime_bar.png` - mean runtime per trial (GSAT vs WalkSAT)
  - `heuristics_time_vs_bestc_scatter.png` - heuristic runs: time vs #clauses satisfied

# Learning Outcomes
- Gained understanding of how complete vs. heurstic SAT algorithms differ. DPLL guarantees correctness but scales poorly, while GSAT and WalkSAT trade completeness for speed.
- Observed that heuristic algorithms can quickly find near-optimal solutions even when DPLL times out.
- Observed the effect of problem size (uf20 -> uf75) on algorithm runtime and quality of solutions.
- Developed skills in collecting and analyzing performance data. Used consistent CPU timing, CSV logging, and visualization via graphs.
- Gained an understanding of the importance of parameter tuning (flip limit, noise, timeout) to balance speed vs. accuracy in heuristic searches.
