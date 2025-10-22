import os
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


# -------------------------- Evaluation helpers --------------------------


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

    # write CSV results
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

            print(f'Processing {p} ...')
            res = run_on_file(p, timeout=args.timeout)
            writer.writerow(res)
            print('  ->', res)


if __name__ == '__main__':
    main()
