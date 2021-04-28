import numpy as np

from MultWeights import MultiplicativeWeights


class MultiplicativeWeights2(MultiplicativeWeights):

    def __init__(self, beta = 0.5):
        super().__init__(beta=beta)

    def update_weights(self, weights, preds, outcome):
        mean = np.mean(weights)
        idx = np.logical_and(preds != outcome, self.weights >= 0.25*mean)
        weights[idx] *= self.beta
        return weights