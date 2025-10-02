# A* on the Sudoku Cube

## What this is
An A* solver for a Sudoku Cube with experiment running and plotting tools.

## How to run
1) Install dependencies:  
   `python -m pip install -r requirements.txt`
2) Run the program:  
   `python a-star.py`
3) Use the menu:
   - **[1] Shuffle & print** – simple testing function (same as Program 1).
   - **[2] Solve one** – build solved cube -> CW scramble of k moves -> solve using A* with only counterclockwise moves. Prints the "last iteration" metric.
   - **[3] Run experiments** – for each `k=3->k_max`, run N trials, write `astar_results_YYYYMMDD_HHMMSS.csv`.
   - **[4] Analyze & plot** – read a CSV, compute average "nodes expanded in last iteration" per k, save `astar_summary_*.csv` and `astar_plot_*.png`.

## Reproducibility via seeds
- Experiments accept an optional "base seed", and each trial calculates `seed = base + k*1000 + t`.
- "Solve one instance" [3] also accepts a seed.

## Notes
- We scramble with **CW only** and solve **CCW only**.
- If searches start to blow up, there’s a `max_expansions` safety valve in `astar()` (default is `None`).
