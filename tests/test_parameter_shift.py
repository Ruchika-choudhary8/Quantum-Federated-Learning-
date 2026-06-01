import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

from models.quantum_model import (
    circuit,
    n_qubits
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

X = torch.rand(1, 4)

y = torch.tensor([1.0])

weights = torch.randn(
    n_qubits,
    requires_grad=True
)


output = circuit(
    X[0],
    weights
)

output = torch.stack(output).mean()

loss = (output - y[0]) ** 2

loss.backward()

autograd_grad = (
    weights.grad.detach().clone()
)


# -----------------------------
# Parameter Shift Gradient
# -----------------------------
shift_grad = []

shift = np.pi / 2

base_output = circuit(
    X[0],
    weights.detach()
)
base_output = torch.stack(
    base_output
).mean()

for i in range(n_qubits):

    shifted_plus = weights.detach().clone()
    shifted_minus = weights.detach().clone()

    shifted_plus[i] += shift
    shifted_minus[i] -= shift

    forward_plus = circuit(
        X[0],
        shifted_plus
    )
    forward_plus = torch.stack(
    	forward_plus
    ).mean()
 
    forward_minus = circuit(
        X[0],
        shifted_minus
    )
    forward_minus = torch.stack(
    	forward_minus
    ).mean()

    dfdtheta = (
        forward_plus - forward_minus
    ) / 2

    # Chain rule
    dLdf = 2 * (
        base_output - y[0]
    )

    grad_i = dLdf * dfdtheta

    shift_grad.append(grad_i)

shift_grad = torch.stack(shift_grad)

cosine_similarity = nn.functional.cosine_similarity(
    autograd_grad,
    shift_grad,
    dim=0
)

print("\nAutograd Gradient:\n")
print(autograd_grad)

print("\nParameter Shift Gradient:\n")
print(shift_grad)

print("\nCosine Similarity:\n")
print(cosine_similarity.item())