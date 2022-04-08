"""
Script for solving a single problem of 6 that we consider in this work.

The line | means that the algorithm found the solution better than Greedy.
The dot . means that the algorithm found the solution worse than Greedy.

Examples:
    python solve_multiple_problems.py help
    python solve_multiple_problems.py seed=1234 max_time=300 number_of_repetitions=3
"""

from AntColony import args_to_dict
import sys

from solve_single_problem import solve_single_problem

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', 'help', '-help', '--help'):
        print(__doc__)
        raise ValueError("printed help and aborted")

    seed = None
    max_time = None
    number_of_repetitions = 11

    input_params = args_to_dict(sys.argv[1:], globals(), print=print)
    globals().update(input_params)

    for i in range(1, 7):  # i \in {1,2,3,4,5,6}
        solve_single_problem(problem=i, seed=seed, max_time=max_time, number_of_repetitions=number_of_repetitions)

    print("\n\nMultiple problems solved")