import numpy as np
from utils import softmax, one_hot

class Client:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self, global_weights, lr=0.1):
        X = self.X
        y = self.y

        logits = X @ global_weights
        probs = softmax(logits)

        y_onehot = one_hot(y)

        grad = (X.T @ (probs - y_onehot)) / len(X)

        updated_weights = global_weights - lr * grad
        return updated_weights