# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as itr
from collections.abc import Iterable
from solution import Solution
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import random as rnd

import configuration as cfg
import os


class TabuSearch:
    __max_iteration = 100
    __number_of_top_candidates = 5
    __estimator = BaseEstimator()
    __best_solution = None
    __train_X = None
    __train_y = None
    __test_X = None
    __test_y = None
    __random_state = 0
    __params = None
    __start_solution = None
    __tabu_list = dict() # {'parameter_name': iterations_tabu}

    @property
    def best_solution(self):
        return self.__best_solution

    def __init__(self, estimator: BaseEstimator, params: dict, data_X: np.array, data_y: np.array, max_iteration: int = 100,
                 number_of_top_candidates: int = 5, random_state: int = 0):
        self.__max_iteration = max_iteration
        self.__number_of_top_candidates = number_of_top_candidates
        self.__estimator = estimator
        self.__random_state = random_state

        self.__train_X, self.__test_X, self.__train_y, self.__test_y = train_test_split(data_X, data_y, test_size=0.2,
                                                                                        random_state=self.__random_state)

        start_params = dict()

        for key in params.keys():
            if not hasattr(self.__estimator, key):
                raise TypeError(f'Estimator don\'t has attribute \'{key}\'.')
            if not isinstance(params[key], Iterable):
                raise TypeError(f'Parameter \'{key}\' must be iterable')

            start_params[key] = params[key][0]

        self.__params = params

        start_solution = Solution(start_params)
        self.__calculate_performance(start_solution)
        self.__start_solution = start_solution

    @staticmethod
    def __product_dict(p: dict):
        keys = p.keys()
        vals = p.values()
        for instance in itr.product(*vals):
            yield dict(zip(keys, instance))

    def __set_params_of_estimator(self, params: dict) -> BaseEstimator:
        estimator = self.__estimator

        for key in params.keys():
            if not isinstance(params[key], Iterable) or isinstance(params[key], str):
                value = params[key]
            else:
                value = params[key][0]

            setattr(estimator, key, value)

        return estimator

    def __calculate_performance(self, s: Solution):
        estimator = self.__set_params_of_estimator(s.parameters)
        estimator.fit(self.__train_X, self.__train_y)
        pred_y = estimator.predict(self.__test_X)

        s.performance = accuracy_score(self.__test_y, pred_y)

        return s.performance

    def __generate_neighborhood_solutions(self, s: Solution):
        params = self.__params.copy()
        neighborhood = list()

        neighbor_params = self.__product_dict(p=params)

        for tabu_item in self.__tabu_list:
            del params[tabu_item]

        rnd_param = rnd.choice(list(params.keys()))
        for val in params[rnd_param]:
            new_params = s.parameters.copy()
            new_params[rnd_param] = val
            new_solution = Solution(new_params)

            self.__calculate_performance(new_solution)

            neighborhood.append(new_solution)

        return neighborhood

    def __find_best_solution(self, solutions):
        perf = []
        for solution in solutions:
            perf.append(solution.performance)

        i_max = np.argmax(perf)

        return solutions[i_max]

    def perform_search(self):

        iteration = 1

        solution = self.__start_solution

        while iteration <= self.__max_iteration:
            nh_solutions = self.__generate_neighborhood_solutions(solution)
            best_nh_solution = self.__find_best_solution(nh_solutions)

            if best_nh_solution != solution:
                solution = best_nh_solution
            elif 'aspiration criteria' == True:
                #TODO: set apiration criteria
                solution = best_nh_solution
            else:
                # find best not tabu solution in the neighborhood
                # s_c = s_nt
                solution = best_nh_solution

            self.__tabu_list.append(solution)

            iteration += 1

            for tabu_item in self.__tabu_list:
                if self.__tabu_list[tabu_item] == 1:
                    del self.__tabu_list[tabu_item]
                else:
                    self.__tabu_list[tabu_item] -= 1

        __best_solution = solution


if __name__ == '__main__':
    # start the tabu search
    random_number = 5

    # traffic volume
    traffic_y = np.loadtxt(os.path.join(cfg.default.traffic_data, 'traffic_volume_y'), delimiter=',')
    traffic_X = np.loadtxt(os.path.join(cfg.default.traffic_data, 'traffic_volume_X'), delimiter=',')

    ridge_estimator = Ridge()

    params_ridge = {'alpha': np.arange(0, 1, 0.1),
                    'solver': ['auto', 'svd', 'cholesky']}

    search = TabuSearch(estimator=ridge_estimator, params=params_ridge, data_X=traffic_X, data_y=traffic_y,
                        max_iteration=10, number_of_top_candidates=5)

    search.perform_search()
