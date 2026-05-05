import numpy as np
from model import quantum_model

def loss_fn(weights, x, y):
    pred = quantum_model(weights, x)
    return (pred - y) ** 2