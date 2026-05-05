import numpy as np
from optimizer import parameter_shift

class Client:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self, global_weights, lr=0.1, epochs=2):
        weights = global_weights.copy()

        for _ in range(epochs):
            for x, y in zip(self.X, self.y):
                grad = parameter_shift(weights, x, y)
                weights = weights - lr * grad

        return weights