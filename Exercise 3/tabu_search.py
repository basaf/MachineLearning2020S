# -*- coding: utf-8 -*-

import itertools as itr
import os
import random as rnd
from collections.abc import Iterable, Callable
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import configuration as cfg
import preprocessing
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
    __tabu_list = dict()
    __error_metric = None
    __search_space = None
    __history_best_solution = list()
    __history_solution = list()

    @property
    def best_solution(self):
        return self.__best_solution

    @property
    def history_solution(self):
        return self.__history_solution

    @property
    def history_best_solution(self):
        return self.__history_best_solution

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

        self.__tabu_list = dict()

        # generate all possible solutions
        search_space = dict()
        for estimator in self.__estimators.keys():
            search_space[estimator] = list(self.__product_dict(self.__params[estimator]))
            self.__tabu_list[estimator] = dict()

        self.__search_space = search_space

        self.__start_solution = self.__get_rnd_solution(list(self.__estimators.keys()))

        self.__history_best_solution = list()
        self.__history_solution = list()

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
        while is_in_tabu_list:
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

            print(
                f'iteration: {iteration}; no improve: {no_improvement_iteration}; solution: {solution}; best solution: {self.__best_solution}')

            self.__history_best_solution.append(self.__best_solution)
            self.__history_solution.append(solution)

            iteration += 1


if __name__ == '__main__':
    # start the tabu search
    random_number = 10

    communities_x, communities_y = preprocessing.preprocess_communities()
    traffic_x, traffic_y = preprocessing.preprocess_traffic()
    estate_x, estate_y = preprocessing.preprocess_real_estate()
    student_x, student_y = preprocessing.preprocess_student()

    datasets = {'communities': {'x': communities_x, 'y': communities_y},
                'traffic': {'x': traffic_x, 'y': traffic_y},
                'estate': {'x': estate_x, 'y': estate_y},
                'student': {'x': student_x, 'y': student_y}}

    ridge_estimator = Ridge()
    knn = KNeighborsRegressor(n_jobs=-1)
    r_forest = RandomForestRegressor(random_state=random_number, n_jobs=-1)
    nn = MLPRegressor(random_state=random_number) # , max_iter=500
    svm = SVR()

    params_ridge = {'alpha': np.arange(0, 1, 0.1),
                    'fit_intercept': [True, False],
                    'normalize': [True, False]}

    params_knn = {'n_neighbors': range(1, 101),
                  'weights': ['uniform', 'distance'],
                  'p': [1, 2]}

    params_r_forest = {'n_estimators': [20, 50, 100, 200],
                       'criterion': ['mse', 'mae'],
                       'min_samples_split': range(2, 10),
                       'min_samples_leaf': range(1, 10),
                       'bootstrap': [True, False]}

    params_nn = {'hidden_layer_sizes': range(1, 10),
                 'activation': ['identity', 'logistic', 'tanh', 'relu']}

    params_svm = {'kernel': ['linear', 'poly', 'rbf'],
                  'degree': range(2, 10),
                  'C': np.arange(1e-2, 100.)}

    estimators = {'ridge': ridge_estimator,
                  'knn': knn,
                  'random forest': r_forest,
                  'neural network': nn,
                  'svm': svm}

    parameters = {'ridge': params_ridge,
                  'knn': params_knn,
                  'random forest': params_r_forest,
                  'neural network': params_nn,
                  'svm': params_svm}

    marker_cycle = ['o', '+', 'x', '1', '2']

    for dataset_key in datasets:
        print(f'start {dataset_key}')
        dataset = datasets[dataset_key]

        search = TabuSearch(estimators=estimators, params=parameters, error_metric=r2_score,
                            data_X=dataset['x'], data_y=dataset['y'], max_iteration=100, max_no_improvement_iteration=20,
                            number_of_tabu_iterations=5, random_state=random_number)

        search.perform_search()

        history_solution = search.history_solution
        history_best_solution = search.history_best_solution

        fig_sol = plt.figure(figsize=(6, 10))
        for estimator_idx, estimator_key in enumerate(estimators.keys()):
            iteration = [x_i for x_i, x in enumerate(history_solution) if x.estimator_key == estimator_key]
            performance = [x.performance for x in history_solution if x.estimator_key == estimator_key]

            plt.scatter(x=iteration, y=performance, marker=marker_cycle[estimator_idx])

        plt.legend(estimators.keys())
        plt.xticks(range(len(history_solution)), [str(x.parameters) for x in history_solution], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.default.traffic_figures, f'{dataset_key}_history_solution.png'),
                    format='png', dpi=300)
        plt.close(fig_sol)

        fig_best = plt.figure(figsize=(6, 10))
        for estimator_idx, estimator_key in enumerate(estimators.keys()):
            iteration = [x_i for x_i, x in enumerate(history_best_solution) if x.estimator_key == estimator_key]
            performance = [x.performance for x in history_best_solution if x.estimator_key == estimator_key]

            plt.scatter(x=iteration, y=performance, marker=marker_cycle[estimator_idx])

        plt.legend(estimators.keys())
        plt.xticks(range(len(history_best_solution)), [str(x.parameters) for x in history_best_solution], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.default.traffic_figures, f'{dataset_key}_history_best_solution.png'),
                    format='png', dpi=300)
        plt.close(fig_best)

        print(f'end {dataset_key}')
