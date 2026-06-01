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

    X = torch.rand(64, 4)

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

    gradient_values = []

    for epoch in range(5):

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        grad = (
            model.weights.grad
            .detach()
            .cpu()
            .numpy()
        )

        gradient_values.extend(
            grad
        )

        optimizer.step()

    gradient_values = np.array(
        gradient_values
    )

    print("\nNoise p =", p)

    print(
        "Mean Gradient Magnitude:",
        np.mean(
            np.abs(
                gradient_values
            )
        )
    )

    print(
        "Gradient Variance:",
        np.var(
            gradient_values
        )
    )


run_experiment(0.0)

run_experiment(0.1)