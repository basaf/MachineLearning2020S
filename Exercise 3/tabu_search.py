# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as itr
from solution import Solution

# %% Init
#TODO: configure value ranges


# %% Helper functions
def generate_rnd_solution():
    return Solution({})


def calculate_performance(s: Solution):
    return 0


def generate_neighborhood_solutions(s: Solution):
    return []


def find_best_solution(solutions):
    perf = calculate_performance(solutions)
    i_max = np.argmax(perf)

    return solutions[i_max]


# %% tabu search
tabu_list = []
searching = True
number_of_top_candidates = 5
max_iterations = 100
iteration = 1

solution = generate_rnd_solution()
solution_performance = calculate_performance(solution)

while iteration <= max_iterations:
    nh_solutions = generate_neighborhood_solutions(solution)
    best_nh_solution = find_best_solution(nh_solutions)

    if best_nh_solution is not solution:
        solution = best_nh_solution
    elif 'aspiration criteria' is True:
        #TODO: set apiration criteria
        solution = best_nh_solution
    else:
        # find best not tabu solution in the neighborhood
        # s_c = s_nt
        solution = best_nh_solution

    tabu_list.append(solution)

    iteration += 1



    
