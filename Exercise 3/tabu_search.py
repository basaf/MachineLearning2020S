# -*- coding: utf-8 -*-

import itertools as itr
import os
import random as rnd
from collections.abc import Iterable, Callable
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import configuration as cfg
from solution import Solution


class TabuSearch:
    __max_iteration = 100
    __max_no_improvement_iteration = 20
    __number_of_tabu_interations = 5
    __estimators = dict()
    __best_solution = None
    __train_X = None
    __train_y = None
    __test_X = None
    __test_y = None
    __random_state = 0
    __params = dict()
    __start_solution = None
    __tabu_list = dict()  # {'parameter_name': iterations_tabu}
    __error_metric = None
    __search_space = None

    @property
    def best_solution(self):
        return self.__best_solution

    def __init__(self, estimators: dict, params: dict, error_metric: Callable,
                 data_X: np.array, data_y: np.array, max_iteration: int = 100, max_no_improvement_iteration: int = 20,
                 number_of_tabu_iterations: int = 5, random_state: int = 0):
        self.__max_iteration = max_iteration
        self.__max_no_improvement_iteration = max_no_improvement_iteration
        self.__number_of_tabu_interations = number_of_tabu_iterations

        if not isinstance(estimators, dict) and not any(isinstance(est, BaseEstimator) for est in estimators.values()):
            raise TypeError(
                f'The parameter \'estimators\' must be a dictionary, key=name of estimator, value must be type of \'BaseEstimator\'')

        self.__estimators = estimators

        self.__random_state = random_state
        self.__error_metric = error_metric

        self.__train_X, self.__test_X, self.__train_y, self.__test_y = train_test_split(data_X, data_y, test_size=0.2,
                                                                                        random_state=self.__random_state)
        if not isinstance(params, dict) and not any(isinstance(parm, dict) for parm in params.values()):
            raise TypeError(
                f'The parameter \'params\' must be a dictionary, key=name of estimator, value must be a dictionary of the parameters')

        for estimator in params.keys():
            for key in params[estimator].keys():
                if not hasattr(self.__estimators[estimator], key):
                    raise TypeError(f'Estimator don\'t has attribute \'{key}\'.')
                if not isinstance(params[estimator][key], Iterable):
                    raise TypeError(f'Parameter \'{key}\' must be iterable')

        self.__params = params

        # generate all possible solutions
        search_space = dict()
        for estimator in self.__estimators.keys():
            search_space[estimator] = list(self.__product_dict(self.__params[estimator]))
            self.__tabu_list[estimator] = dict()

        self.__search_space = search_space

        self.__start_solution = self.__get_rnd_solution(list(self.__estimators.keys()))

    def __get_rnd_solution(self, estimator_key: Union[str, list]) -> Solution:
        search_space = self.__search_space

        if not isinstance(estimator_key, str) and not (
                isinstance(estimator_key, list) and any(isinstance(est, str) for est in estimator_key)):
            raise AttributeError(f'Attribute \'estimator_key\' must be either a string or a list of strings')

        if isinstance(estimator_key, list):
            rnd_estimator = rnd.choice(range(len(estimator_key)))
            estimator_key = estimator_key[rnd_estimator]

        params_estimator = search_space[estimator_key]
        tabu_list = list(self.__tabu_list[estimator_key].keys())

        # ensures that the loop is run at least once
        is_in_tabu_list = True
        while(is_in_tabu_list):
            rnd_param = rnd.choice(range(len(params_estimator)))
            is_in_tabu_list = rnd_param in tabu_list

        solution = Solution(parameters=params_estimator[rnd_param], estimator_key=estimator_key, id=rnd_param)
        self.__calculate_performance(solution)

        return solution

    @staticmethod
    def __product_dict(p: dict):
        keys = p.keys()
        vals = p.values()
        for instance in itr.product(*vals):
            yield dict(zip(keys, instance))

    @staticmethod
    def __set_params_of_estimator(estimator: BaseEstimator, params: dict) -> BaseEstimator:
        for key in params.keys():
            if not isinstance(params[key], Iterable) or isinstance(params[key], str):
                value = params[key]
            else:
                value = params[key][0]

            setattr(estimator, key, value)

        return estimator

    def __calculate_performance(self, s: Solution):
        estimator = self.__estimators[s.estimator_key]
        estimator = self.__set_params_of_estimator(estimator, s.parameters)
        estimator.fit(self.__train_X, self.__train_y)
        pred_y = estimator.predict(self.__test_X)

        s.performance = self.__error_metric(self.__test_y, pred_y)

        return s.performance

    def __generate_neighborhood_solutions(self, s: Solution) -> (list, list):
        estimator_key = s.estimator_key
        id_solution = s.id
        estimator_params = self.__search_space[estimator_key]
        tabu_list = self.__tabu_list[estimator_key]
        size_of_solution_space = len(estimator_params)

        if id_solution < 2:
            start = 0
        else:
            start = id_solution - 2

        if id_solution > size_of_solution_space - 3:
            stop = size_of_solution_space
        else:
            stop = id_solution + 3

        neighborhood = list()
        tabu_neighborhood = list()

        # to remove the initial solution add ' - set([id_solution])'
        for id_n in set(range(start, stop)):
            sol_n = Solution(parameters=estimator_params[id_n], estimator_key=estimator_key, id=id_n)
            self.__calculate_performance(sol_n)

            if id_n in tabu_list:
                tabu_neighborhood.append(sol_n)
            else:
                neighborhood.append(sol_n)

        return neighborhood, tabu_neighborhood

    @staticmethod
    def __find_best_solution(solutions: list):

        perf = np.asarray([sol.performance for sol in solutions])

        if len(perf) == 0:
            return None
        else:
            return solutions[np.argmax(a=perf)]

    def perform_search(self):
        iteration = 1
        no_improvement_iteration = 1

        solution = self.__start_solution
        self.__best_solution = solution

        while iteration <= self.__max_iteration and no_improvement_iteration <= self.__max_no_improvement_iteration:
            nh_solutions, nh_tabu_solutions = self.__generate_neighborhood_solutions(solution)

            if len(nh_solutions) == 0:
                print('no neighborhood found')
                break

            best_nh_solution = self.__find_best_solution(nh_solutions)
            best_nh_tabu_solution = self.__find_best_solution(nh_tabu_solutions)

            if best_nh_solution != solution and solution < best_nh_solution:
                solution = best_nh_solution
            else:
                if best_nh_tabu_solution is not None and self.__best_solution < best_nh_tabu_solution:
                    solution = best_nh_tabu_solution
                else:
                    # switch algorithm
                    estimator_keys = list(set(self.__estimators) - {solution.estimator_key})
                    solution = self.__get_rnd_solution(estimator_key=estimator_keys)

            tabu_list = self.__tabu_list[solution.estimator_key]

            for tabu_item_key in list(tabu_list):
                if tabu_list[tabu_item_key] == 1:
                    del tabu_list[tabu_item_key]
                else:
                    tabu_list[tabu_item_key] -= 1

            tabu_list[solution.id] = self.__number_of_tabu_interations

            self.__tabu_list[solution.estimator_key] = tabu_list

            if self.__best_solution < solution:
                self.__best_solution = solution
                no_improvement_iteration = 1
            else:
                no_improvement_iteration += 1

            print(f'iteration: {iteration}; solution: {solution}; best solution: {self.__best_solution}')

            iteration += 1


if __name__ == '__main__':
    # start the tabu search
    random_number = 5

    # traffic volume
    traffic_y = np.loadtxt(os.path.join(cfg.default.traffic_data, 'traffic_volume_y'), delimiter=',')
    traffic_X = np.loadtxt(os.path.join(cfg.default.traffic_data, 'traffic_volume_X'), delimiter=',')

    ridge_estimator = Ridge()
    knn = KNeighborsRegressor(n_jobs=-1)

    params_ridge = {'alpha': np.arange(0, 1, 0.1),
                    'fit_intercept': [True, False],
                    'normalize': [True, False]}

    params_knn = {'n_neighbors': range(1, 10, 1),
                  'weights': ['uniform', 'distance'],
                  'leaf_size': range(1, 5, 1)}

    estimators = {'ridge': ridge_estimator,
                  'knn': knn}
    parameters = {'ridge': params_ridge,
                  'knn': params_knn}

    print('start')

    search = TabuSearch(estimators=estimators, params=parameters, error_metric=explained_variance_score,
                        data_X=traffic_X, data_y=traffic_y, max_iteration=100, max_no_improvement_iteration=20,
                        number_of_tabu_iterations=5)

    search.perform_search()

    print('end')
