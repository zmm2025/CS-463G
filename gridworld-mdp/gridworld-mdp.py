"""
CS 463G Markov Decision Processes Assignment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable, Mapping, Optional

State = Tuple[int, int] # (x, y) with x, y in {0, ..., 5}
Action = str            # 'U', 'D', 'L', 'R'

# ----------------------
# GLOBAL MDP PARAMETERS
# ----------------------

GAMMA: float = 0.95

P_INTENDED: float = 0.70
P_REVERSE: float = 0.15
P_STAY: float = 0.15

ACTIONS: List[Action] = ["U", "D", "L", "R"]

# Movement direction for each action
ACTION_DIRS: Dict[Action, Tuple[int, int]] = {
    "U": (0, 1),
    "D": (0, -1),
    "L": (-1, 0),
    "R": (1, 0),
}

# Reverse of each action (for backwards movement)
REVERSE_ACTION: Dict[Action, Action] = {
    "U": "D",
    "D": "U",
    "L": "R",
    "R": "L",
}

# Arrow symbols for visualization
ACTION_ARROWS: Dict[Action, str] = {
    "U": "^",
    "D": "v",
    "L": "<",
    "R": ">",
}

# --------------
# GRIDWORLD MDP
# --------------

@dataclass
class GridWorldMDP:
    width: int
    height: int
    rewards: Mapping[State, float]
    terminal_states: Iterable[State]
    gamma: float = GAMMA

    entry_state: State = (0, 0)
    wumpus_state: Optional[State] = None

    def __post_init__(self) -> None:
        self.terminal_states = set(self.terminal_states)

    # ------------ Basic helpers ------------

    def all_states(self) -> List[State]:
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    def in_bounds(self, state: State) -> bool:
        x, y = state
        return 0 <= x < self.width and 0 <= y < self.height

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_states

    def reward(self, state: State) -> float:
        return float(self.rewards.get(state, 0.0))

    # ------------ Transition model ------------

    def move(self, state: State, action: Action) -> State:
        """Single move; if it would hit a wall, stay put."""
        x, y = state
        dx, dy = ACTION_DIRS[action]
        new_x, new_y = x + dx, y + dy
        if (0 <= new_x < self.width) and (0 <= new_y < self.height):
            return (new_x, new_y)
        return state

    def transition_probs(self, state: State, action: Action) -> Dict[State, float]:
        """
        P(state' | state, action) for the manual-transmission car:

            0.70 -> intended direction
            0.15 -> reverse direction
            0.15 -> stay in place

        If state is terminal, stay there with probability 1.
        """
        if self.is_terminal(state):
            return {state: 1.0}

        intended = self.move(state, action)
        reverse = self.move(state, REVERSE_ACTION[action])
        stay = state

        probs: Dict[State, float] = {}

        def add_prob(state: State, prob: float) -> None:
            probs[state] = probs.get(state, 0.0) + prob

        add_prob(intended, P_INTENDED)
        add_prob(reverse, P_REVERSE)
        add_prob(stay, P_STAY)

        return probs

# ---------------------------------------------------------------------------
# Starting grids
# Coordinates are (x, y): x = column (0 left -> 5 right), y = row (0 bottom -> 5 top).
# ---------------------------------------------------------------------------

ENTRY_STATE: State = (0, 0)

# ---- GRID 1 ----
GRID1_REWARDS: Dict[State, float] = {
    (4, 4): 1800.0,
    # everything else defaults to 0.0
}

GRID1_WUMPUS: Optional[State] = (4, 2)

def make_grid1_mdp() -> GridWorldMDP:
    terminal_states = set()

    # Treasure cell is terminal
    terminal_states.add((4, 4))

    # Wumpus is also terminal with reward 0
    if GRID1_WUMPUS is not None:
        terminal_states.add(GRID1_WUMPUS)

    return GridWorldMDP(
        width           = 6,
        height          = 6,
        rewards         = GRID1_REWARDS,
        terminal_states = terminal_states,
        gamma           = GAMMA,
        entry_state     = ENTRY_STATE,
        wumpus_state    = GRID1_WUMPUS,
    )

# ---- GRID 2 ----
GRID2_REWARDS: Dict[State, float] = {
    (0, 2): 20.0,
    (0, 3): -5.0,
    (1, 5): 10.0,
    (4, 4): 1700.0,
    (5, 1): -20.0,
}

GRID2_WUMPUS: Optional[State] = (4, 2)

def make_grid2_mdp() -> GridWorldMDP:
    terminal_states = {
        (0, 2),
        (0, 3),
        (1, 5),
        (4, 4),
        (5, 1),
    }

    if GRID2_WUMPUS is not None:
        terminal_states.add(GRID2_WUMPUS)

    return GridWorldMDP(
        width           = 6,
        height          = 6,
        rewards         = GRID2_REWARDS,
        terminal_states = terminal_states,
        gamma           = GAMMA,
        entry_state     = ENTRY_STATE,
        wumpus_state    = GRID2_WUMPUS,
    )

# -------------------------------
# Algorithms (to be implemented)
# -------------------------------

def run_value_iteration(
    mdp: GridWorldMDP,
    horizon: int,
) -> Tuple[Dict[State, float], Dict[State, Action]]:
    """
    Run Value Iteration on the given MDP for the specified number of iterations (time horizon).

    Behavior:
        - Initialize V(s) = 0 for all states.
        - For t in 1->horizon:
              For each non-terminal state s:
                  V_new(s) = max_a sigma_{s'} P(s' | s, a) * [ R(s') + mdp.gamma * V(s') ]
              Terminal states keep their value fixed: V(s) = R(s).
        - After the loop, derive the greedy policy:
              policy[s] = argmax_a sigma_{s'} P(s' | s, a) * [ R(s') + mdp.gamma * V(s') ]
          for non-terminal states only.

    Returns
    -------
    V : dict
        Mapping from state -> value estimate.
    policy : dict
        Mapping from state -> greedy action according to V.
    """
    pass


def run_policy_iteration(
    mdp: GridWorldMDP,
    horizon: int,
) -> Tuple[Dict[State, float], Dict[State, Action]]:
    """
    Run (Modified) Policy Iteration on the given MDP.

    'horizon' specifies the number of Bellman update sweeps performed
    during each Policy Evaluation step. This means we do NOT solve the
    full system of equations for V^pi; instead, we approximate it with
    'horizon' iterations of value updates under the current policy.

    Behavior:
        1. Initialize a policy pi(s) arbitrarily for all non-terminal states (e.g., always "U").
        2. Loop until the policy is stable (or a max number of outer iterations):
             a) Policy Evaluation:
                    Initialize V(s) = 0 (or keep from previous iteration),
                    then repeat 'horizon' times:
                        For each state s:
                            If s is terminal:
                                V(s) = R(s)  (do not change)
                            Else:
                                V(s) = sigma_{s'} P(s' | s, pi(s)) * [ R(s') + γ * V(s') ]
             b) Policy Improvement:
                    For each non-terminal state s:
                        Find best_action = argmax_a sigma_{s'} P(s' | s, a) * [ R(s') + γ * V(s') ]
                        If best_action != pi(s), update pi(s) and mark policy_changed = True.
             c) If policy_changed is False, stop.

    Returns
    -------
    V : dict
        Mapping from state -> value under the final policy.
    policy : dict
        Mapping from state -> final policy's action in each state.
    """
    pass

# ----------------------
# Visualization helpers
# ----------------------

def empty_grid(width: int, height: int, fill: float = 0.0) -> List[List[float]]:
    """Create a 2D list [y][x] with given fill value."""
    return [[fill for _ in range(width)] for _ in range(height)]


def values_to_grid(
    V: Mapping[State, float],
    width: int,
    height: int,
) -> List[List[float]]:
    grid = empty_grid(width, height, fill=0.0)
    for (x, y), v in V.items():
        grid[y][x] = v
    return grid


def policy_to_grid(
    policy: Mapping[State, Action],
    width: int,
    height: int,
    terminal_states: Iterable[State],
) -> List[List[str]]:
    grid = [["." for _ in range(width)] for _ in range(height)]
    terminals = set(terminal_states)
    for y in range(height):
        for x in range(width):
            state = (x, y)
            if state in terminals:
                grid[y][x] = "T"  # terminal
            elif state in policy:
                grid[y][x] = ACTION_ARROWS[policy[state]]
    return grid


def print_value_grid(grid: List[List[float]]) -> None:
    """
    Print values in a grid with y = height-1 at the top.
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    for y in reversed(range(height)):
        row = " | ".join(f"{grid[y][x]:7.2f}" for x in range(width))
        print(row)
    print()


def print_policy_grid(grid: List[List[str]]) -> None:
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    for y in reversed(range(height)):
        row = " | ".join(f" {grid[y][x]:>2} " for x in range(width))
        print(row)
    print()


# ---------------------------------------------------------------------------
# Example usage for testing (after algorithm implementation is done)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # NOTE: DON'T change the transition model or rewards.
    # Just implement run_value_iteration and run_policy_iteration,
    # then run them with horizons 50 and 100 for both grids.
    
    mdp1 = make_grid1_mdp()
    mdp2 = make_grid2_mdp()

    print("Grid 1 states:", len(mdp1.all_states()))
    print("Grid 2 states:", len(mdp2.all_states()))

    # V1_50, pi1_50 = run_value_iteration(mdp1, horizon=50)
    # grid_vals = values_to_grid(V1_50, mdp1.width, mdp1.height)
    # print_value_grid(grid_vals)
