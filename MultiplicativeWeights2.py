from MultWeights import MultiplicativeWeights


class MultiplicativeWeights2(MultiplicativeWeights):

    def __init__(self, beta = 0.5):
        super().__init__(beta=beta)

    def update_weights(self, weights, preds, outcome):
        pass #TODO