import numpy as np
from utils import softmax

class Model:
    def __init__(self, input_size=784, num_classes=10):
        self.weights = np.zeros((input_size, num_classes))

    def forward(self, X):
        return softmax(X @ self.weights)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)