"""
Script for solving a single problem of 6 that we consider in this work.

Examples:
    python solve_single_problem.py help
    python solve_single_problem.py problem=1 # small_1
    python solve_single_problem.py problem=6 # big_2
    python solve_single_problem.py problem=4 seed=1234 max_time = 300
"""

import numpy as np
from AntColony import *
import sys

np.seterr(divide='ignore', invalid='ignore')  # this warning is not important, it is served in package

def solve_single_problem(problem, seed, max_time):
    if problem is None:
        raise Exception("Problem number should be provided. See python solve_single_problem.py help")
    if max_time is None:
        max_time = 5

    problem_number = problem
    rng_seed = seed

    print("Optimizing problem no. {} with seed = {} with max_time = {}".format(problem_number, rng_seed, max_time))

    rng = np.random.default_rng(rng_seed)

    problem_strings = [None,
                       "./../../data/augerat/A-n32-k05.xml",
                       "./../../data/augerat/A-n44-k06.xml",
                       "./../../data/augerat/A-n60-k09.xml",
                       "./../../data/augerat/A-n69-k09.xml",
                       "./../../data/uchoa/X-n101-k25.xml",
                       "./../../data/uchoa/X-n120-k6.xml"]

    max_cars_list = [None, 7, 6, 9, 10, 28, 8]
    s_max_list = [None, 205, 285, 223, 223, 1750, 2641.2]

    problem = Problem(problem_strings[problem_number])
    max_cars = max_cars_list[problem_number]
    s_max = s_max_list[problem_number]

    # Greedy
    print("\nGreedy")
    greedy = Greedy(print_warnings=False)
    greedy.set_problem(s_max, max_cars, problem)
    greedy.optimize()
    save_to_file(greedy, "./logs/problem_{}/greedy.log".format(problem_number))

    # Basic
    print("\nBasic")
    for i in range(11):
        seed_this_iter = rng.integers(1, 1e6)

        antColony_basic = AntColony(print_warnings=False)
        antColony_basic.set_problem(s_max, max_cars, problem)
        antColony_basic.optimize(max_iter=10000, print_progress=False, max_time=max_time, rng_seed=seed_this_iter)

        save_to_file(antColony_basic, "./logs/problem_{}/basic_{:.0f}_{}.log".format(problem_number,
                                                                             antColony_basic.best_cost,
                                                                             seed_this_iter))

        dot_or_line = "|" if antColony_basic.best_cost < greedy.best_cost else "."
        print(dot_or_line, end="", flush=True)  # with flush Python prints without waiting for the EOL sign

    # Reduced
    print("\n\nReduced")
    for i in range(11):
        seed_this_iter = rng.integers(1, 1e6)

        antColony_reduced = AntColony_Reduced(print_warnings=False)
        antColony_reduced.set_problem(s_max, max_cars, problem)
        antColony_reduced.optimize(max_iter=10000, print_progress=False, max_time=max_time, rng_seed=seed_this_iter)
        save_to_file(antColony_reduced, "./logs/problem_{}/reduced_{:.0f}_{}.log".format(problem_number,
                                                                                         antColony_reduced.best_cost,
                                                                                         seed_this_iter))
        dot_or_line = "|" if antColony_reduced.best_cost < greedy.best_cost else "."
        print(dot_or_line, end="", flush=True)

    # Divided
    print("\n\nDivided")
    for i in range(11):
        seed_this_iter = rng.integers(1, 1e6)

        antColony_divided = AntColony_Divided(print_warnings=False)
        antColony_divided.set_problem(s_max, max_cars, problem)
        antColony_divided.optimize(max_iter=10000, print_progress=False, max_time=max_time, rng_seed=seed_this_iter)
        save_to_file(antColony_divided, "./logs/problem_{}/divided_{:.0f}_{}.log".format(problem_number,
                                                                                 antColony_divided.best_cost,
                                                                                 seed_this_iter))

        dot_or_line = "|" if antColony_divided.best_cost < greedy.best_cost else "."
        print(dot_or_line, end="", flush=True)

    print("\n\nSingle problem solved")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', 'help', '-help', '--help'):
        print(__doc__)
        raise ValueError("printed help and aborted")

    # Problem number should be overwritten
    problem = None
    seed = None
    max_time = None

    input_params = args_to_dict(sys.argv[1:], globals(), print=print)
    globals().update(input_params)

    solve_single_problem(problem, seed, max_time)