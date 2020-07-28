# -*- coding: utf-8 -*-


class Solution:
    __parameters = dict()
    __performance = 0.0
    __estimator_key = ''
    __id = -1

    def __init__(self, parameters: dict, estimator_key: str, id: int):
        self.__parameters = parameters
        self.__estimator_key = estimator_key
        self.__id = id

    @property
    def id(self):
        return self.__id

    @property
    def estimator_key(self):
        return self.__estimator_key

    @property
    def performance(self):
        return self.__performance

    @performance.setter
    def performance(self, value):
        self.__performance = value

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, dictionary):
        if isinstance(dictionary, dict):
            self.__parameters = dictionary
        else:
            raise AttributeError(f'{dictionary.__class__.__name__} is a invalid attribute.')

    def __eq__(self, other):
        # x==y
        if not isinstance(other, Solution):
            raise AttributeError('can only compared to a solution')
        else:
            return self.__parameters == other.parameters and self.__estimator_key == other.estimator_key

    def __lt__(self, other):
        # x<y
        if not isinstance(other, Solution):
            raise AttributeError('can only compared to a solution')
        else:
            return self.__performance < other.performance

    def __le__(self, other):
        # x<=y
        if not isinstance(other, Solution):
            raise AttributeError('can only compared to a solution')
        else:
            return self.__performance <= other.performance

    def __gt__(self, other):
        # x>y
        if not isinstance(other, Solution):
            raise AttributeError('can only compared to a solution')
        else:
            return self.__performance > other.performance

    def __ge__(self, other):
        # x>=y
        if not isinstance(other, Solution):
            raise AttributeError('can only compared to a solution')
        else:
            return self.__performance >= other.performance

    def __repr__(self):
        return repr((self.parameters, self.performance, self.estimator_key, self.id))
