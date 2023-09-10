from numpy import argmax
import numpy as np


def comparator_factory(kind, delta=1e-3):
    if kind == 'regression' or kind == 'reg':
        return RegressionComparator(delta)
    elif kind == 'classification' or kind == 'class':
        return ClassificationComparator()
    raise ValueError('unknown kind: ' + kind)


class Comparator:
    def compare(self, expected, actual):
        pass


class RegressionComparator(Comparator):
    def __init__(self, delta):
        if delta < 0 or delta > 0.5:
            raise ValueError('bad delta value')
        self.__delta = delta

    def compare_scalar(self, expected, actual):
        return abs(expected - actual) <= self.__delta

    def compare(self, expected, actual):
        if np.isscalar(expected) and np.isscalar(actual):
            return self.compare_scalar(expected, actual)
        if np.isscalar(expected):
            if len(actual) > 1:
                return False
            return self.compare_scalar(expected, actual[0])
        elif np.isscalar(actual):
            if len(expected) > 1:
                return False
            return self.compare_scalar(expected[0], actual)
        n = len(expected)
        if n != len(actual):
            return False
        res = True
        for i in range(0, n):
            res = res and self.compare_scalar(expected[i], actual[i])
        return res


class ClassificationComparator(Comparator):
    def compare(self, expected, actual):
        if np.isscalar(expected) and np.isscalar(actual):
            return expected == round(actual)
        if np.isscalar(expected):
            if len(actual) == 1:
                return expected == round(actual[0])
            return expected == argmax(actual)
        elif np.isscalar(actual):
            if len(expected) == 1:
                return expected[0] == round(actual)
            return argmax(expected) == round(actual)
        if len(expected) != len(actual):
            return False
        return argmax(expected) == argmax(actual)
