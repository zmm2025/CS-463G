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
# Algorithms (VI and PI)
# -------------------------------

#Value Iteration
def run_value_iteration(
    mdp: GridWorldMDP,
    horizon: int,
) -> Tuple[Dict[State, float], Dict[State, Action]]:

    V: Dict[State, float] = {s: 0.0 for s in mdp.all_states()}

    # Terminal states always equal their reward
    for s in mdp.terminal_states:
        V[s] = mdp.reward(s)

    for _ in range(horizon):
        V_new = V.copy()
        for s in mdp.all_states():
            if mdp.is_terminal(s):
                V_new[s] = mdp.reward(s)
                continue

            best_value = float("-inf")
            for a in ACTIONS:
                total = 0.0
                for s2, prob in mdp.transition_probs(s, a).items():
                    total += prob * (mdp.reward(s2) + mdp.gamma * V[s2])
                best_value = max(best_value, total)

            V_new[s] = best_value

        V = V_new

    # ---- Greedy policy extraction ----
    policy: Dict[State, Action] = {}

    for s in mdp.all_states():
        if mdp.is_terminal(s):
            continue

        best_action = None
        best_value = float("-inf")

        for a in ACTIONS:
            total = 0.0
            for s2, prob in mdp.transition_probs(s, a).items():
                total += prob * (mdp.reward(s2) + mdp.gamma * V[s2])

            if total > best_value:
                best_value = total
                best_action = a

        policy[s] = best_action

    return V, policy

# Policy Iteration
def run_policy_iteration(
    mdp: GridWorldMDP,
    horizon: int,
) -> Tuple[Dict[State, float], Dict[State, Action]]:

    states = mdp.all_states()

    # Initialize arbitrary policy (choose "U" for non-terminals)
    policy: Dict[State, Action] = {
        s: "U" for s in states if not mdp.is_terminal(s)
    }

    # Initialize value function
    V: Dict[State, float] = {s: 0.0 for s in states}
    for s in mdp.terminal_states:
        V[s] = mdp.reward(s)

    policy_stable = False

    while not policy_stable:
        # --------------------
        # Policy Evaluation
        # --------------------
        for _ in range(horizon):
            V_new = V.copy()
            for s in states:
                if mdp.is_terminal(s):
                    V_new[s] = mdp.reward(s)
                    continue

                a = policy[s]
                total = 0.0
                for s2, prob in mdp.transition_probs(s, a).items():
                    total += prob * (mdp.reward(s2) + mdp.gamma * V[s2])

                V_new[s] = total

            V = V_new

        # --------------------
        # Policy Improvement
        # --------------------
        policy_stable = True

        for s in states:
            if mdp.is_terminal(s):
                continue

            old_action = policy[s]

            best_action = None
            best_value = float("-inf")

            for a in ACTIONS:
                total = 0.0
                for s2, prob in mdp.transition_probs(s, a).items():
                    total += prob * (mdp.reward(s2) + mdp.gamma * V[s2])

                if total > best_value:
                    best_value = total
                    best_action = a

            policy[s] = best_action

            if best_action != old_action:
                policy_stable = False

    return V, policy

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


# --------------------------
# Example usage for testing
# --------------------------

if __name__ == "__main__":
    
    mdp1 = make_grid1_mdp()
    mdp2 = make_grid2_mdp()

    print("Grid 1 states:", len(mdp1.all_states()))
    print("Grid 2 states:", len(mdp2.all_states()))

    V1_50, pi1_50 = run_value_iteration(mdp1, horizon=50)
    grid_vals = values_to_grid(V1_50, mdp1.width, mdp1.height)
    print_value_grid(grid_vals)

    PV1_50, PP1_50 = run_policy_iteration(mdp1, horizon=50)
    grid_policy = policy_to_grid(PP1_50, mdp1.width, mdp1.height, mdp1.terminal_states)
    print_policy_grid(grid_policy)

    V1_100, pi1_100 = run_value_iteration(mdp1, horizon=100)
    grid_vals = values_to_grid(V1_100, mdp1.width, mdp1.height)
    print_value_grid(grid_vals)
    
    PV1_100, PP1_100 = run_policy_iteration(mdp1, horizon=100)
    grid_policy = policy_to_grid(PP1_100, mdp1.width, mdp1.height, mdp1.terminal_states)
    print_policy_grid(grid_policy)

    V2_50, pi2_50 = run_value_iteration(mdp2, horizon=50)
    grid_vals = values_to_grid(V2_50, mdp2.width, mdp2.height)
    print_value_grid(grid_vals)

    PV2_50, PP2_50 = run_policy_iteration(mdp2, horizon=50)
    grid_policy = policy_to_grid(PP2_50, mdp2.width, mdp2.height, mdp2.terminal_states)
    print_policy_grid(grid_policy)

    V2_100, pi2_100 = run_value_iteration(mdp2, horizon=100)
    grid_vals = values_to_grid(V2_100, mdp2.width, mdp2.height)
    print_value_grid(grid_vals)

    PV2_100, PP2_100 = run_policy_iteration(mdp2, horizon=100)
    grid_policy = policy_to_grid(PP1_100, mdp2.width, mdp2.height, mdp2.terminal_states)
    print_policy_grid(grid_policy)