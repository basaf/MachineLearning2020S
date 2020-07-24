# -*- coding: utf-8 -*-


class Solution:
    __parameters = dict()
    __performance = 0.0

    def __init__(self, parameters: dict):
        self.__parameters = parameters

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
            return False
        else:
            return self.__parameters == other.parameters

    def __lt__(self, other):
        # x<y
        if not isinstance(other, Solution):
            return False
        else:
            return self.__performance < other.performance

    def __le__(self, other):
        # x<=y
        if not isinstance(other, Solution):
            return False
        else:
            return self.__performance <= other.performance

    def __gt__(self, other):
        # x>y
        if not isinstance(other, Solution):
            return False
        else:
            return self.__performance > other.performance

    def __ge__(self, other):
        # x>=y
        if not isinstance(other, Solution):
            return False
        else:
            return self.__performance >= other.performance
