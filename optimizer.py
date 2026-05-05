import numpy as np
from loss import loss_fn

def parameter_shift(weights, x, y):
    grad = np.zeros_like(weights)

    for i in range(len(weights)):
        shift = np.zeros_like(weights)
        shift[i] = np.pi / 2

        loss_plus = loss_fn(weights + shift, x, y)
        loss_minus = loss_fn(weights - shift, x, y)

        grad[i] = (loss_plus - loss_minus) / 2

    return grad