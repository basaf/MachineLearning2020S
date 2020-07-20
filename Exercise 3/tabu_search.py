# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Init
#TODO: configure value ranges


# %% Helper functions
def generate_rnd_solution():
    return []


def calculate_performance(s):
    return 0


def generate_neighborhood_solutions(s):
    return []


def find_best_solution(solutions):
    perf = calculate_performance(solutions)
    i_max = np.argmax(perf)

    return solutions[i_max]


# %% tabu search
tabu_list = []
searching = True

solution = generate_rnd_solution()
solution_performance = calculate_performance(solution)

while(searching):
    nh_solutions = generate_neighborhood_solutions(solution)
    best_nh_solution = find_best_solution(nh_solutions)

    if best_nh_solution is not solution:
        solution = best_nh_solution
    elif 'aspiration criteria' is True:
        solution = best_nh_solution
    else:
        # find best not tabu solution in the neighborhood
        # s_c = s_nt

    tabu_list.append(solution)



    
