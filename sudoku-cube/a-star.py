from collections import defaultdict
from dataclasses import dataclass
import heapq
import math # For rounding up heuristic calculations
import random # For randomizing cube moves
from typing import Dict, Iterator, List, Optional, Set, Tuple

"""
Sudoku Cube Simulator
------------------------------------
This script simulates a 3x3x3 cube with numbered faces, allows random shuffling,
and prints the cube in a net layout. The user can specify the number of random moves.
"""

Move = tuple[str, str]

class Face:
    TOP: str    = 'top'
    BOTTOM: str = 'bottom'
    LEFT: str   = 'left'
    RIGHT: str  = 'right'
    FRONT: str  = 'front'
    BACK: str   = 'back'

    CW: str = 'clockwise'
    CCW: str = 'counterclockwise'

    ROW: str = 'row'
    COL: str = 'col'

    def __init__(self, cube, side: str, tiles: list):
        self.cube = cube
        self.side = side
        self.tiles = tiles
    
    def __getitem__(self, key):
        return self.tiles[key]
    
    def __setitem__(self, key, value):
        self.tiles[key] = value
    
    def rotate(self, direction: str):
        # Rotate the face itself
        match direction:
            case Face.CW:
                self.tiles = [list(row) for row in zip(*self.tiles[::-1])]
            case Face.CCW:
                self.tiles = [list(row) for row in zip(*self.tiles)][::-1]
            case _:
                raise ValueError("Invalid direction")
        
        
        if self.side == Face.FRONT:
            # F: T (bottom row), R (col 0), D (top row), L (col 2)
            if direction == Face.CW:
                temp = self.cube[Face.TOP][2][:]
                # T->R, R->D, D->L, L->T
                for i in range(3):
                    self.cube[Face.TOP   ][2  ][i  ] = self.cube[Face.LEFT  ][2-i][2  ]
                    self.cube[Face.LEFT  ][2-i][2  ] = self.cube[Face.BOTTOM][0  ][2-i]
                    self.cube[Face.BOTTOM][0  ][2-i] = self.cube[Face.RIGHT ][i  ][0  ]
                    self.cube[Face.RIGHT ][i  ][0  ] = temp[i]
            else:
                temp = self.cube[Face.TOP][2][:]
                for i in range(3):
                    self.cube[Face.TOP   ][2  ][i  ] = self.cube[Face.RIGHT ][i  ][0  ]
                    self.cube[Face.RIGHT ][i  ][0  ] = self.cube[Face.BOTTOM][0  ][2-i]
                    self.cube[Face.BOTTOM][0  ][2-i] = self.cube[Face.LEFT  ][2-i][2  ]
                    self.cube[Face.LEFT  ][2-i][2  ] = temp[i]
        elif self.side == Face.BACK:
            # B: T (top row), L (col 0), D (bottom row), R (col 2)
            if direction == Face.CW:
                temp = self.cube[Face.TOP][0][:]
                for i in range(3):
                    self.cube[Face.TOP   ][0  ][i  ] = self.cube[Face.RIGHT ][i  ][2  ]
                    self.cube[Face.RIGHT ][i  ][2  ] = self.cube[Face.BOTTOM][2  ][2-i]
                    self.cube[Face.BOTTOM][2  ][2-i] = self.cube[Face.LEFT  ][2-i][0  ]
                    self.cube[Face.LEFT  ][2-i][0  ] = temp[i]
            else:
                temp = self.cube[Face.TOP][0][:]
                for i in range(3):
                    self.cube[Face.TOP   ][0  ][i  ] = self.cube[Face.LEFT  ][2-i][0  ]
                    self.cube[Face.LEFT  ][2-i][0  ] = self.cube[Face.BOTTOM][2  ][2-i]
                    self.cube[Face.BOTTOM][2  ][2-i] = self.cube[Face.RIGHT ][i  ][2  ]
                    self.cube[Face.RIGHT ][i  ][2  ] = temp[i]
        elif self.side == Face.TOP:
            # T: B (top row), R (top row), F (top row), L (top row)
            if direction == Face.CW:
                temp = self.cube[Face.BACK][0][:]
                self.cube[Face.BACK  ][0  ] = self.cube[Face.RIGHT ][0  ][:]
                self.cube[Face.RIGHT ][0  ] = self.cube[Face.FRONT ][0  ][:]
                self.cube[Face.FRONT ][0  ] = self.cube[Face.LEFT  ][0  ][:]
                self.cube[Face.LEFT  ][0  ] = temp
            else:
                temp = self.cube[Face.BACK][0][:]
                self.cube[Face.BACK  ][0  ] = self.cube[Face.LEFT  ][0  ][:]
                self.cube[Face.LEFT  ][0  ] = self.cube[Face.FRONT ][0  ][:]
                self.cube[Face.FRONT ][0  ] = self.cube[Face.RIGHT ][0  ][:]
                self.cube[Face.RIGHT ][0  ] = temp
        elif self.side == Face.BOTTOM:
            # D: F (bottom row), R (bottom row), B (bottom row), L (bottom row)
            if direction == Face.CW:
                temp = self.cube[Face.FRONT][2][:]
                self.cube[Face.FRONT ][2  ] = self.cube[Face.RIGHT ][2  ][:]
                self.cube[Face.RIGHT ][2  ] = self.cube[Face.BACK  ][2  ][:]
                self.cube[Face.BACK  ][2  ] = self.cube[Face.LEFT  ][2  ][:]
                self.cube[Face.LEFT  ][2  ] = temp
            else:
                temp = self.cube[Face.FRONT][2][:]
                self.cube[Face.FRONT ][2  ] = self.cube[Face.LEFT  ][2  ][:]
                self.cube[Face.LEFT  ][2  ] = self.cube[Face.BACK  ][2  ][:]
                self.cube[Face.BACK  ][2  ] = self.cube[Face.RIGHT ][2  ][:]
                self.cube[Face.RIGHT ][2  ] = temp
        elif self.side == Face.LEFT:
            # L: T (col 0), F (col 0), D (col 0), B (col 2, reversed)
            if direction == Face.CW:
                temp = [self.cube[Face.TOP][i][0] for i in range(3)]
                for i in range(3):
                    self.cube[Face.TOP   ][i  ][0  ] = self.cube[Face.BACK  ][2-i][2  ]
                    self.cube[Face.BACK  ][2-i][2  ] = self.cube[Face.BOTTOM][i  ][0  ]
                    self.cube[Face.BOTTOM][i  ][0  ] = self.cube[Face.FRONT ][i  ][0  ]
                    self.cube[Face.FRONT ][i  ][0  ] = temp[i]
            else:
                temp = [self.cube[Face.TOP][i][0] for i in range(3)]
                for i in range(3):
                    self.cube[Face.TOP   ][i  ][0  ] = self.cube[Face.FRONT ][i  ][0  ]
                    self.cube[Face.FRONT ][i  ][0  ] = self.cube[Face.BOTTOM][i  ][0  ]
                    self.cube[Face.BOTTOM][i  ][0  ] = self.cube[Face.BACK  ][2-i][2  ]
                    self.cube[Face.BACK  ][2-i][2  ] = temp[i]
        elif self.side == Face.RIGHT:
            # R: T (col 2), B (col 0, reversed), D (col 2), F (col 2)
            if direction == Face.CW:
                temp = [self.cube[Face.TOP][i][2] for i in range(3)]
                for i in range(3):
                    self.cube[Face.TOP   ][i  ][2  ] = self.cube[Face.FRONT ][i  ][2  ]
                    self.cube[Face.FRONT ][i  ][2  ] = self.cube[Face.BOTTOM][i  ][2  ]
                    self.cube[Face.BOTTOM][i  ][2  ] = self.cube[Face.BACK  ][2-i][0  ]
                    self.cube[Face.BACK  ][2-i][0  ] = temp[i]
            else:
                temp = [self.cube[Face.TOP][i][2] for i in range(3)]
                for i in range(3):
                    self.cube[Face.TOP   ][i  ][2  ] = self.cube[Face.BACK  ][2-i][0  ]
                    self.cube[Face.BACK  ][2-i][0  ] = self.cube[Face.BOTTOM][i  ][2  ]
                    self.cube[Face.BOTTOM][i  ][2  ] = self.cube[Face.FRONT ][i  ][2  ]
                    self.cube[Face.FRONT ][i  ][2  ] = temp[i]
        return self

class Cube:
    def __init__(self):
        self.faces = {}
    
    def __getitem__(self, key):
        return self.faces[key]
    
    def __setitem__(self, key, value):
        self.faces[key] = value
    
    def face(self, side: str, tiles: list):
        face = Face(self, side, tiles)
        self.faces[side] = face
        return face
    
    @property
    def heuristic(self) -> int:
        """
        Calculates an admissible heuristic value for a minimum number of turns needed to find a solution.
        Takes the maximum calculation from the two following cases:
            1. Minimum number of turns to fix the side with the most displaced tiles
            2. Minimum number of turns to fix all displaced tiles
        Returns the calculated minimum turns needed.
        """
        MAX_TILE_FIX_PER_FACE = 3 # max num of tiles fixed per turn on a single side
        MAX_TILE_FIX = 12 # max num of tiles fixed per turn
        
        # Create dictionary of all face deficits
        face_deficits = {}
        for side in self.faces:
            face = self.faces[side]
            nums_present = set()
            for tile_row in face:
                nums_present.update(tile_row)
            face_deficit = 9 - len(nums_present)
            face_deficits[side] = face_deficit
        
        # Get max 3-tile deficit fix
        side_deficit = max(face_deficits.values())
        side_turns_needed = math.ceil(side_deficit / MAX_TILE_FIX_PER_FACE)
        
        # Get sum of all deficits' fix
        sum_deficit = sum(face_deficits.values())
        sum_turns_needed = math.ceil(sum_deficit / MAX_TILE_FIX)
        
        # Return max of the two deficit fixes
        min_turns_needed = max(side_turns_needed, sum_turns_needed)
        return min_turns_needed
    
    def shuffle(self, num_moves: int) -> list:
        """
        Randomly shuffle the cube by performing num_moves random quarter-turns.
        Ensures no move is immediately undone (e.g., F then F').
        Returns the list of performed moves.
        """
        face_choices = [Face.TOP, Face.BOTTOM, Face.LEFT, Face.RIGHT, Face.FRONT, Face.BACK]
        direction_choices = [Face.CW, Face.CCW]
        last_move = None
        move_sequence = []

        for _ in range(num_moves):
            while True:
                face = random.choice(face_choices)
                direction = random.choice(direction_choices)

                # Don't undo the last move
                if last_move is None:
                    break
                last_face, last_direction = last_move
                # If same face and opposite direction, skip
                if (face == last_face) and (direction != last_direction):
                    continue

                break

            # Rotate
            self[face].rotate(direction)

            move = (face, direction)
            move_sequence.append(move)
            last_move = move
        
        return move_sequence

    def neighbors(self, skip_inverse_of: tuple[str, str] | None = None) -> Iterator[tuple[tuple[str, str], "Cube"]]:
        """
        Generate an iterable of all legal next states as pairs: ((face, direction), next_cube)
        If skip_inverse_of=(face, dir) is given, omit the move that
        would undo that last step (same side, opposite direction)
        """
        face_choices = [Face.TOP, Face.BOTTOM, Face.LEFT, Face.RIGHT, Face.FRONT, Face.BACK]
        direction_choices = [Face.CW, Face.CCW]

        for face in face_choices:
            for direction in direction_choices:
                if skip_inverse_of is not None:
                    last_face, last_dir = skip_inverse_of
                    if face == last_face and direction != last_dir:
                        continue
                
                next_cube = self.clone()
                next_cube[face].rotate(direction)
                yield (face, direction), next_cube

    def print(self):
        """
        Print the cube as a 2D net with each face as a 3x3 grid.
        Layout:
                [  Top   ]
        [ Left ][ Front  ][ Right ][ Back ]
                [ Bottom ]
        """
        face_rows = [
            [None,            self[Face.TOP],    None,             None],
            [self[Face.LEFT], self[Face.FRONT],  self[Face.RIGHT], self[Face.BACK]],
            [None,            self[Face.BOTTOM], None,             None]
        ]

        # First line
        print("        +-------+                ")

        for face_row_index, face_row in enumerate(face_rows):
            for line_index in range(3):
                # Left border of Left Face
                print(' ' if face_row[0] is None else '|', end="")
                
                for face_index, face in enumerate(face_row):
                    # All 3 tiles, or blanks if no face
                    if face is None:
                        print("       ", end="")
                    else:
                        print(f" {face[line_index][0]} {face[line_index][1]} {face[line_index][2]} ", end="")
                    
                    # Separators between faces
                    if face_index < 3:
                        separator = '|'
                        if (face is None) and (face_row[face_index + 1] is None):
                            separator = ' '
                        print(separator, end="")
                
                # Right border of Right Face
                print(' ' if face_row[-1] is None else '|')
            
            # Separators between face rows
            if face_row_index < 2:
                print("+-------+-------+-------+-------+")
        
        # Last line
        print("        +-------+                ")
    
    def is_solved(self) -> bool:
        """
        Each face is considered solved if it contains all numbers 1-9 exactly once.
        The cube is considered solved if all 6 faces are solved.
        """
        side_list = [Face.TOP, Face.BOTTOM, Face.LEFT, Face.RIGHT, Face.FRONT, Face.BACK]
        for side in side_list:
            face = self[side]
            seen_digits = []
            for row in range(3):
                for col in range(3):
                    seen_digits.append(face[row][col])
            if set(seen_digits) != set(range(1, 10)):
                return False
        return True
    
    def clone(self) -> "Cube":
        """
        Creates a deep copy of a Cube.
        """
        new_cube = Cube()
        for side in self.faces:
            tiles = [[self[side][row][col] for col in range(3)] for row in range(3)]
            new_cube.face(side, tiles)
        return new_cube
    
    def state_key(self) -> tuple:
        """
        Returns an immutable "snapshot" of the cube's state.
        Order matches the cube net layout: TOP, LEFT, FRONT, RIGHT, BACK, BOTTOM.
        """
        side_order = [Face.TOP, Face.LEFT, Face.FRONT, Face.RIGHT, Face.BACK, Face.BOTTOM]
        state_key = []
        for side in side_order:
            face = self[side]
            for row in range(3):
                for col in range(3):
                    state_key.append(face[row][col])
        return tuple(state_key)

@dataclass
class Node:
    """
    A single search node used for the A* algorithm.
    - cube:     The cube configuration at this node
    - g:        Cost so far (measured in quarter-turns from start)
    - move:     Move taken from parent node (None for the start)
    - parent:   Link to the previous node (used for path reconstruction)
    """
    cube: Cube
    g: int
    move: Optional[Move] = None
    parent: Optional["Node"] = None

    def path(self) -> List[Move]:
        """
        Reconstruct the move sequence from the start to this node.
        Ex: [(front, "clockwise"), (right, "counterclockwise"), ...]
        """
        moves: List[Move] = []
        node_cursor: Optional[Node] = self
        while node_cursor is not None and node_cursor.move is not None:
            moves.append(node_cursor.move)
            node_cursor = node_cursor.parent
        moves.reverse()
        return moves

class Frontier:
    """
    Priority queue ordered by (f, g, counter).
    - f = g + h (A*)
    - When f is a tie, g is used as the second key and prefers deeper nodes.
    - Counter breaks any remaining ties deterministically rather than randomly.
    """
    def __init__(self) -> None:
        self.heap: List[Tuple[int, int, int, Node]] = []
        self.counter: int = 0
    
    def push(self, f: int, g: int, node: Node) -> None:
        heapq.heappush(self.heap, (f, g, self.counter, node))
        self.counter += 1
    
    def pop(self) -> Node:
        return heapq.heappop(self.heap)[-1]
    
    def __bool__(self) -> bool:
        return bool(self.heap)
    
    def __len__(self) -> int:
        return len(self.heap)

def opposite_dir(direction: str) -> str:
    return Face.CCW if direction == Face.CW else Face.CW

def apply_random_moves(
    cube: Cube,
    k: int,
    *,
    seed: int | None = None,
    only_cw: bool = True,
    prevent_undo: bool = True,
) -> list[Move]:
    """
    Apply k random quarter-turns to cube in-place and return the list of moves.
    Defaults:
        - only_cw=True: scramble with only clockwise turns
        - prevent_undo=True: avoid reversing the turn of the move just performed
    """
    rng = random.Random(seed)
    faces = [Face.TOP, Face.BOTTOM, Face.LEFT, Face.RIGHT, Face.FRONT, Face.BACK]
    directions = [Face.CW] if only_cw else [Face.CW, Face.CCW]

    moves: list[Move] = []
    last_move: Move | None = None

    for _ in range(k):
        while True:
            face = rng.choice(faces)
            direction = rng.choice(directions)
            if prevent_undo and last_move is not None:
                last_face, last_dir = last_move
                if face == last_face and direction == opposite_dir(last_dir):
                    continue
            break

        cube[face].rotate(direction)
        move_taken: Move = (face, direction)
        moves.append(move_taken)
        last_move = move_taken
    
    return moves

def astar(
    start_cube: Cube,
    *,
    prevent_undo: bool = True,
    max_expansions: Optional[int] = None
) -> tuple[List[Move], dict]:
    """
    A* search using:
        - g = depth (quarter-turns)
        - h = start_cube.heuristic (admissible)
        - f = g + h
    
    Returns (path, stats) where:
        - path: List[Move], i.e. [(face, dir), ...] (empty if no solution found)
        - stats: {
            "solution_length": int | None,
            "nodes_expanded_total": int,
            "nodes_expanded_last_iteration": int | None,
            "f_star": int | None,
            "expansions_by_f": Dict[int, int],
            "visited_size": int,
            "frontier_size": int,
            "pushes": int
          }
    """
    # Frontier ordered by (f, g, counter)
    frontier = Frontier()

    # Start node
    start_node = Node(cube=start_cube, g=0)
    f0 = 0 + start_cube.heuristic
    frontier.push(f=f0, g=0, node=start_node)    

    # Best g seen per state; visited set
    g_best: Dict[tuple, int] = { start_cube.state_key(): 0 }
    visited_states: Set[tuple] = set()

    # Other variables
    expansions_by_f: Dict[int, int] = defaultdict(int)
    total_expanded = 0
    pushes = 1 # we pushed the start

    while frontier:
        cur = frontier.pop()
        key = cur.cube.state_key()

        # Skip if we've already permanently visited this exact state
        if key in visited_states:
            continue

        # Count this expansion at tis current f (for last-iteration metric)
        f_cur = cur.g + cur.cube.heuristic
        expansions_by_f[f_cur] += 1
        total_expanded += 1

        # Test if goal was reached
        if cur.cube.is_solved():
            f_star = cur.g # cost of an optimal solution
            path = cur.path()
            stats = {
                "solution_length": cur.g,
                "nodes_expanded_total": total_expanded,
                "nodes_expanded_last_iteration": expansions_by_f.get(f_star, 0),
                "f_star": f_star,
                "expansions_by_f": dict(expansions_by_f),
                "visited_size": len(visited_states),
                "frontier_size": len(frontier),
                "pushes": pushes,
            }
            return path, stats
        
        # Mark as visited and expand neighbors
        visited_states.add(key)

        move_to_skip = cur.move if prevent_undo else None
        for move, next_cube in cur.cube.neighbors(skip_inverse_of=move_to_skip):
            next_key = next_cube.state_key()
            tentative_g = cur.g + 1

            # If we found a cheaper way to reach this state, push & update it
            if tentative_g < g_best.get(next_key, float("inf")):
                g_best[next_key] = tentative_g
                f_child = tentative_g + next_cube.heuristic
                child = Node(cube=next_cube, g=tentative_g, move=move, parent=cur)
                frontier.push(f=f_child, g=tentative_g, node=child)
                pushes += 1
        
        # Optional safety check to stop if search starts to explode
        if max_expansions is not None and total_expanded >= max_expansions:
            break
    
    # No solution found (or stopped by max_expansions)
    stats = {
        "solution_length": None,
        "nodes_expanded_total": total_expanded,
        "nodes_expanded_last_iteration": None,
        "f_star": None,
        "expansions_by_f": dict(expansions_by_f),
        "visited_size": len(visited_states),
        "frontier_size": len(frontier),
        "pushes": pushes,
    }
    return [], stats

def main() -> None:
    # Main interactive loop: repeatedly ask for number of random moves and shuffle the cube
    
    # Initialize and print initial cube
    cube = Cube()
    tiles = {
        cube.face(Face.TOP,    [[8,1,3], [4,6,7], [2,9,5]]),
        cube.face(Face.BOTTOM, [[1,2,8], [5,3,9], [7,4,6]]),
        cube.face(Face.LEFT,   [[7,1,8], [2,4,6], [9,3,5]]),
        cube.face(Face.RIGHT,  [[4,6,3], [7,5,9], [1,2,8]]),
        cube.face(Face.FRONT,  [[9,5,2], [3,8,1], [6,7,4]]),
        cube.face(Face.BACK,   [[9,5,2], [3,8,1], [6,7,4]]),
    }
    print("Initial cube:")
    cube.print()

    while True:
        try:
            num_moves = int(input("Enter number of random moves to perform: "))
        except ValueError:
            print("Invalid input. Exiting.")
            break
        
        moves = cube.shuffle(num_moves)
        print(f"\nAfter {num_moves} random moves:")
        cube.print()
        print("\nMoves performed:")
        print(moves)
        print(f"Heuristic value: {cube.heuristic}")


if __name__ == "__main__":
    main()
