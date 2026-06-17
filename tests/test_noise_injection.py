import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.noisy_quantum_model import NoisyQuantumModel


def run_experiment(p):

    X = torch.rand(64, 8)

    y = torch.randint(
        0,
        2,
        (64,)
    ).float()

    model = NoisyQuantumModel(
        p=p
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01
    )

    criterion = nn.MSELoss()

    batch_gradients = []

    batch_size = 8

    for epoch in range(5):

         permutation = torch.randperm(64)

         for i in range(0, 64, batch_size):

             idx = permutation[i:i + batch_size]

             X_batch = X[idx]
             y_batch = y[idx]

             optimizer.zero_grad()

             outputs = model(X_batch)

             loss = criterion(
                 outputs,
                 y_batch
             )

             loss.backward()
             if p > 0:
                  noise_std = 9 * p * (
                       model.weights.grad.std().item()
                       + 1e-6
                  )

                  with torch.no_grad():

                       model.weights.grad += (
                            noise_std *
                            torch.randn_like(
                                 model.weights.grad
                            )
                       )

             grad = (
                  model.weights.grad
                  .detach()
                  .cpu()
                  .numpy()
             )

             batch_gradients.append(
                 grad.copy()
             )

             optimizer.step()

    batch_gradients = np.array(
        batch_gradients
    )

    mean_grad_mag = np.mean(
        np.abs(batch_gradients)
    )

    gradient_norms = np.linalg.norm(
        batch_gradients,
        axis=1
    )

    grad_variance = np.mean(
        np.std(batch_gradients, axis=0)
    )

    print("\nNoise p =", p)

    print(
        "Mean Gradient Magnitude:",
        mean_grad_mag        
    )

    print(
        "Gradient Variance:",
        grad_variance
    )


run_experiment(0.0)

run_experiment(0.1)
