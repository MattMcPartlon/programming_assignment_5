from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


def scale_weights(weights, beta=0.5):
    weights /= np.max(weights)
    if np.min(weights) < 2 ** (-20):
        weights *= beta
    return weights


def get_weighted_majority(weights, preds):
    """
    implement this method
    :param weights: weights for each expert
    :param preds: expert's predictions
    :return: a value + or - 1 indicating the majority opinion of the experts
    """
    return np.sign(2 * int(np.sign(np.sum(weights * preds))) - 1)


def get_outcome(data: Dict[str, Any], t: int):
    assert 0 < t < len(data['open'])
    if data['close'][t - 1] > data['open'][t]:
        return -1
    return 1


def get_reward(data, t, decision):
    outcome = get_outcome(data, t)
    abs_rew = abs(data['close'][t - 1] - data['open'][t])
    return outcome * decision * abs_rew


def get_data_up_to_t(data: Dict[str, Any], t: int):
    return {key: data[key][:t] for key in data}


def get_data_from_t(data: Dict[str, Any], t: int):
    return {key: data[key][t:] for key in data}


def get_data_in_range(data, start, end):
    temp = get_data_up_to_t(data, end)
    temp = get_data_from_t(temp, start)
    return temp


def get_mistakes(data, start_day, decisions):
    T = len(data['open']) - start_day
    ground_truth = np.array([get_outcome(data, start_day + t) for t in range(T)])
    decisions = np.array(decisions)
    return len(ground_truth[ground_truth != decisions])


def get_rewards(data, start_day, decisions):
    T = len(data['open']) - start_day
    return sum([get_reward(data, start_day + t, decisions[t]) for t in range(T)])


class MultiplicativeWeights(ABC):

    def __init__(self, beta=0.5):
        self.beta = beta
        self.experts = []

    def add_experts(self, *experts):
        self.experts+= list(*experts)

    def get_predictions(self, data):
        return np.array([e.predict(data) for e in self.experts])

    @abstractmethod
    def update_weights(self, weights, preds, outcome):
        #TODO: Implement different weight update schemes for each variant
        pass

    def alg(self, data, start_day=20):
        weights = np.ones(len(self.experts))
        expert_mistakes = np.zeros(len(self.experts))
        T = len(data['open']) - start_day
        decisions = []

        for t in range(start_day, start_day + T):
            # get the data for days 1...start_day + t - 1
            data_to_t = get_data_up_to_t(data, t)
            # have experts make predictions based on this data,
            # and get the weighted opinion
            #Get predictions from each expert
            preds = None #TODO
            #Get the weighted majority opinion of the experts
            decision = None #TODO
            # reveal the ground truth
            outcome = get_outcome(data, t)
            # track decisions
            decisions.append(decision)
            #update the expert's weights
            weights = None #TODO
            weights = scale_weights(weights)
            #update the number of mistakes made by each expert
            #TODO

        reward = get_rewards(data, start_day, decisions)
        mistakes = get_mistakes(data, start_day, decisions)
        return decisions, expert_mistakes, mistakes, reward
