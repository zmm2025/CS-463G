import os
import random
import time
import csv
import argparse
from typing import List, Tuple, Dict


# -------------------------- DIMACS parsing --------------------------


def parse_dimacs(path: str) -> Tuple[int, int, List[List[int]]]:
    """Parse a DIMACS CNF file.

    Returns:
        (num_vars, num_clauses, clauses)
    - clauses is a list of clauses, each clause is a list of ints (literals).
    - Literals use the standard DIMACS convention: positive for x, negative for ~x.
    """
    num_vars = 0
    num_clauses = 0
    clauses: List[List[int]] = []
    current_clause: List[int] = []

    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('c') or line.startswith('%'):
                # comment or end marker
                continue
            if line.startswith('p'):
                parts = line.split()
                if len(parts) >= 4 and parts[1] == 'cnf':
                    try:
                        num_vars = int(parts[2])
                        num_clauses = int(parts[3])
                    except ValueError:
                        pass
                continue

            # clause tokens may span lines; 0 terminates a clause
            for tok in line.split():
                if tok == '%':
                    break
                if tok == '0':
                    clauses.append(current_clause)
                    current_clause = []
                    continue
                try:
                    lit = int(tok)
                except ValueError:
                    continue
                current_clause.append(lit)

    if current_clause:
        # if file ended without trailing 0
        clauses.append(current_clause)

    return num_vars, num_clauses, clauses


# -------------------------- Helper functions --------------------------

def random_assignment(n_vars: int, rng: random.Random) -> list[bool]:
    # 1-indexed assignment for convenicne; index 0 unused
    return [False] + [rng.choice((False, True)) for _ in range(n_vars)]

def clause_satisfied(clause, assign) -> bool:
    for lit in clause:
        v = abs(lit)
        val = assign[v]
        if (lit > 0 and val) or (lit < 0 and not val):
            return True
    return False

def satisfied_count(clauses, assign) -> int:
    return sum(1 for cl in clauses if clause_satisfied(cl, assign))

def flip(assign, v):
    assign[v] = not assign[v]

def evaluate_partial(clauses: List[List[int]], assignment: Dict[int, bool]) -> Tuple[int, int]:
    """Evaluate clauses under a partial assignment.

    Returns:
        (satisfied_count, undetermined_count)
    - A clause is satisfied if any literal is True under assignment.
    - A clause is undetermined if at least one variable in it is unassigned and no literal is True yet.
    - Otherwise the clause is falsified under the partial assignment.
    """
    sat = 0
    undet = 0

    for clause in clauses:
        clause_satisfied = False
        clause_undet = False
        for lit in clause:
            var = abs(lit)
            if var not in assignment:
                # this variable is not assigned yet: clause may still be satisfied later
                clause_undet = True
                # don't break here because another literal in the same clause could already be True
                continue
            val = assignment[var]
            lit_true = val if lit > 0 else (not val)
            if lit_true:
                clause_satisfied = True
                break
        if clause_satisfied:
            sat += 1
        elif clause_undet:
            undet += 1
        else:
            # fully assigned and falsified
            pass

    return sat, undet


# -------------------------- DPLL Max-SAT search --------------------------


class OptimalFound(Exception):
    """Raised internally to abort the search when an optimal solution is found."""


def maxsat_dpll(clauses: List[List[int]], num_vars: int, timeout: float) -> Tuple[int, Dict[int, bool], float]:
    """Simple (unoptimized) DPLL-style search for Max-SAT.

    Contract:
    - Input: clauses (list of list of ints), num_vars (declared number of variables), timeout (seconds)
    - Output: (best_c, best_assignment, cpu_time)
      - best_c: maximum number of clauses satisfied found within the timeout
      - best_assignment: a (partial or full) assignment achieving best_c (dictionary var->bool)
      - cpu_time: CPU time consumed (seconds)

    This implementation is intentionally straightforward: it explores assignments recursively
    and uses a conservative upper bound (sat + undet) to prune branches.
    """
    start = time.process_time()
    best_c = -1
    best_assign: Dict[int, bool] = {}
    total_clauses = len(clauses)

    # variable ordering: 1..num_vars
    vars_order = list(range(1, num_vars + 1))

    def search(idx: int, assignment: Dict[int, bool]):
        nonlocal best_c, best_assign

        # check timeout
        if time.process_time() - start > timeout:
            raise TimeoutError()

        sat, undet = evaluate_partial(clauses, assignment)
        # upper bound on satisfiable clauses under this partial assignment
        upper = sat + undet
        if upper <= best_c:
            return

        # if we've assigned all variables, update best
        if idx > num_vars:
            if sat > best_c:
                best_c = sat
                best_assign = assignment.copy()
                # optimal possible when all clauses are satisfied
                if best_c == total_clauses:
                    raise OptimalFound()
            return

        var = vars_order[idx - 1]

        # branch: True then False
        assignment[var] = True
        search(idx + 1, assignment)
        assignment[var] = False
        search(idx + 1, assignment)
        del assignment[var]

    try:
        search(1, {})
    except TimeoutError:
        # normal: timeout reached
        pass
    except OptimalFound:
        # found a solution satisfying all clauses
        pass

    cpu_time = time.process_time() - start
    return best_c, best_assign, cpu_time


# -------------------------- GSAT --------------------------


def run_gsat(clauses, n_vars, *, max_flips=10000, noise=0.10, rng=None):
    """
    Very small GSAT: epsilon-greedy random flip, otherwise greedy best flip.
    Returns (best_c, best_assign).
    """
    rng = rng or random.Random()
    assign = random_assignment(n_vars, rng)
    best_c = satisfied_count(clauses, assign)
    best_assign = assign[:]

    for _ in range(max_flips):
        # exit early if fully satisfied
        if best_c == len(clauses):
            return best_c, best_assign[:]
        
        # epsilon step: flip a random variable
        if rng.random() < noise:
            v = rng.randint(1, n_vars)
            flip(assign, v)
            sc = satisfied_count(clauses, assign)
            if sc >= best_c:
                best_c, best_assign = sc, assign[:]
            else:
                flip(assign, v) # revert if it got worse
            continue

        # greedy step: choose var with maximal score gain
        current = satisfied_count(clauses, assign)
        gains = []
        for v in range(1, n_vars + 1):
            flip(assign, v)
            sc = satisfied_count(clauses, assign)
            gains.append((sc - current, v))
            flip(assign, v)
        max_gain = max(gains, key=lambda t: t[0])[0]
        candidates = [v for g, v in gains if g == max_gain]
        v = rng.choice(candidates)
        flip(assign, v)

        # track best so far
        sc = satisfied_count(clauses, assign)
        if sc > best_c:
            best_c, best_assign = sc, assign[:]
    
    return best_c, best_assign

def run_gsat_trials(path, *, trials=10, seed=1, max_flips=10000, noise=0.10):
    num_vars, num_clauses, clauses = parse_dimacs(path)
    rows = []
    for i in range(trials):
        rng = random.Random(seed + i)
        t0 = time.perf_counter()
        best_c, _ = run_gsat(clauses, num_vars, max_flips=max_flips, noise=noise, rng=rng)
        dt = time.perf_counter() - t0
        rows.append({
            'file': os.path.basename(path),
            'path': path,
            'algo': 'gsat',
            'seed': seed + i,
            'num_vars': num_vars,
            'num_clauses': num_clauses,
            'best_c': best_c,
            'cpu_time': f'{dt:.6f}',
        })
    return rows


# -------------------------- WalkSAT --------------------------


def run_walksat(clauses, n_vars, *, max_flips=10000, p=0.5, rng=None):
    """
    Minimal WalkSAT: pick a random unsatisfied clause; with probability p, flip a random var in it,
    otherwise flip the var in it that maximizes satisfied clauses.
    Returns (best_c, best_assign).
    """
    rng = rng or random.Random()
    assign = random_assignment(n_vars, rng)
    best_c = satisfied_count(clauses, assign)
    best_assign = assign[:]

    def unsat_clauses():
        return [cl for cl in clauses if not clause_satisfied(cl, assign)]

    for _ in range(max_flips):
        if best_c == len(clauses):
            return best_c, best_assign[:]

        unsat = unsat_clauses()
        if not unsat:
            return best_c, best_assign[:]

        clause = rng.choice(unsat)

        # random walk with probability p
        if rng.random() < p:
            v = abs(rng.choice(clause))
            flip(assign, v)
        else:
            # greedy among vars in the chosen clause
            vars_in_clause = {abs(lit) for lit in clause}
            current_satisfied = satisfied_count(clauses, assign)
            gains = []
            for v in vars_in_clause:
                flip(assign, v)
                sc = satisfied_count(clauses, assign)
                gains.append((sc - current_satisfied, v))
                flip(assign, v)
            max_gain = max(gains, key=lambda t: t[0])[0]
            candidates = [v for gain, v in gains if gain == max_gain]
            v = rng.choice(candidates)
            flip(assign, v)

        sc = satisfied_count(clauses, assign)
        if sc > best_c:
            best_c, best_assign = sc, assign[:]

    return best_c, best_assign

def run_walksat_trials(path, *, trials=10, seed=1, max_flips=10000, p=0.5):
    num_vars, num_clauses, clauses = parse_dimacs(path)
    rows = []
    for i in range(trials):
        rng = random.Random(seed + i)
        t0 = time.perf_counter()
        best_c, _ = run_walksat(clauses, num_vars, max_flips=max_flips, p=p, rng=rng)
        dt = time.perf_counter() - t0
        rows.append({
            'file': os.path.basename(path),
            'path': path,
            'algo': 'walksat',
            'seed': seed + i,
            'num_vars': num_vars,
            'num_clauses': num_clauses,
            'best_c': best_c,
            'cpu_time': f'{dt:.6f}',
        })
    return rows


# -------------------------- Runner / CLI --------------------------


def run_on_file(path: str, timeout: float = 30.0) -> Dict:
    """Run maxsat_dpll on a single file and return a result dict for CSV writing."""
    num_vars, num_clauses, clauses = parse_dimacs(path)
    best_c, best_assign, cpu_time = maxsat_dpll(clauses, num_vars, timeout)
    return {
        'file': os.path.basename(path),
        'path': path,
        'num_vars': num_vars,
        'num_clauses': num_clauses,
        'best_c': best_c,
        'cpu_time': cpu_time,
        'best_assignment_size': len(best_assign) if best_assign else 0,
    }


def discover_cnf_files(input_path: str) -> List[str]:
    """Return a sorted list of .cnf files under input_path (recursively if directory)."""
    files: List[str] = []
    if os.path.isdir(input_path):
        for root, _, names in os.walk(input_path):
            for n in names:
                if n.lower().endswith('.cnf'):
                    files.append(os.path.join(root, n))
    else:
        files = [input_path]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description='DPLL Max-SAT runner')
    parser.add_argument('--algo', choices=['dpll', 'gsat', 'walksat'], default='dpll', help='Algorithm to run')
    parser.add_argument('--trials', type=int, default=10, help='GSAT: number of trials per file')
    parser.add_argument('--seed', type=int, default=1, help='GSAT: base RNG seed')
    parser.add_argument('--max-flips', type=int, default=10000, help='GSAT: flips per trial')
    parser.add_argument('--noise', type=float, default=0.10, help='GSAT: epsilon noise [0,1]')
    parser.add_argument('--p', type=float, default=0.5, help='WalkSAT: random-walk probability [0,1]')
    parser.add_argument('input', nargs='?', default=None,
                        help='file or folder to process (.cnf). If omitted, defaults to PA3_Benchmarks or cwd.')
    parser.add_argument('--timeout', type=float, default=1.0, help='CPU time limit per formula (seconds)')
    parser.add_argument('--out', default='results.csv', help='CSV output file')

    args = parser.parse_args()

    # resolve default input if none provided
    input_arg = args.input
    if input_arg is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_bench = os.path.join(script_dir, 'PA3_Benchmarks')
        input_arg = default_bench if os.path.isdir(default_bench) else os.getcwd()
        print(f"No input provided; defaulting to '{input_arg}'")

    paths = discover_cnf_files(input_arg)

    # GSAT
    if args.algo == 'gsat':
        with open(args.out, 'w', newline='') as csvfile:
            fieldnames = ['file', 'path', 'algo', 'seed', 'num_vars', 'num_clauses', 'best_c', 'cpu_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for p in paths:
                if os.path.basename(p).startswith("._") or "__MACOSX" in p:
                    continue
                num_vars, num_clauses, clauses = parse_dimacs(p)
                if num_vars == 0 or num_clauses == 0 or not clauses:
                    continue
                print(f'[gsat] {p} ...')
                for row in run_gsat_trials(
                    p,
                    trials=args.trials,
                    seed=args.seed,
                    max_flips=args.max_flips,
                    noise=args.noise
                ):
                    writer.writerow(row)
        return
    
    # WalkSAT
    if args.algo == 'walksat':
        with open(args.out, 'w', newline='') as csvfile:
            fieldnames = ['file', 'path', 'algo', 'seed', 'num_vars', 'num_clauses', 'best_c', 'cpu_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for p in paths:
                if os.path.basename(p).startswith("._") or "__MACOSX" in p:
                    continue
                num_vars, num_clauses, clauses = parse_dimacs(p)
                if num_vars == 0 or num_clauses == 0 or not clauses:
                    continue
                print(f'[walksat] {p} ...')
                for row in run_walksat_trials(
                    p,
                    trials=args.trials,
                    seed=args.seed,
                    max_flips=args.max_flips,
                    p=args.p
                ):
                    writer.writerow(row)
        return

    # DPLL
    with open(args.out, 'w', newline='') as csvfile:
        fieldnames = ['file', 'path', 'num_vars', 'num_clauses', 'best_c', 'cpu_time', 'best_assignment_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for p in paths:
            # Skip MacOS junk files and empty parses
            if os.path.basename(p).startswith("._") or "__MACOSX" in p:
                continue
            num_vars, num_clauses, clauses = parse_dimacs(p)
            if num_vars == 0 or num_clauses == 0 or not clauses:
                continue

            print(f'[dpll] {p} ...')
            res = run_on_file(p, timeout=args.timeout)
            writer.writerow(res)
            print('  ->', res)


if __name__ == '__main__':
    main()
