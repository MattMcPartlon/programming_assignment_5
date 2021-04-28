from abc import ABC, abstractmethod

import numpy as np


class Expert(ABC):

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass


class MeanReversion(Expert):
    def __init__(self, window=5, sign=1):
        self.window = window
        self.sign = sign

    def predict(self, data):
        temp = data['close']
        window = self.window
        if temp[-1] > np.mean(temp[:-1][-window:]):
            return self.sign
        return -self.sign


class ExpectationExpert(Expert):
    def __init__(self, window=5, sign=1):
        self.window = window
        self.sign = sign

    def predict(self, data):
        window = self.window
        window = min(window + 1, len(data) - 1)
        daily_changes = zip(data['open'][-window + 1:], data['close'][-window:-1])
        pos_prob = np.array([np.sign(b - a) for a, b in daily_changes])
        if len(pos_prob[pos_prob > 0]) > len(pos_prob[pos_prob < 0]):
            return self.sign
        return -self.sign


class YesterdaysNews(Expert):

    def predict(self, data):
        if len(data) < 1:
            return 1
        if data['close'][-1] > data['open'][-1]:
            return 1
        return -1


class VolumeWeightedCloseTrend(Expert):

    def __init__(self, window=5):
        self.window = window

    def predict(self, data):
        if len(data) < 1:
            return 1
        window = self.window
        vols = data['volume'][:window]
        total_vol = sum(vols)
        if sum(data['close'][-window] * vols / total_vol) > data['close'][-1]:
            return -1
        return 1


def moving_average(a, n=6):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
