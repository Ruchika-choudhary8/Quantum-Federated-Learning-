import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch
import torch.nn as nn
import torch.optim as optim

from models.quantum_model import QuantumModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Tiny random dataset
X = torch.rand(8, 4)

# Binary labels
y = torch.randint(
    0,
    2,
    (8,)
).float()

model = QuantumModel().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

# Forward pass
outputs = model(X)

loss = criterion(outputs, y)

# Backward pass
loss.backward()

print("\nGradient Magnitudes:\n")

for name, param in model.named_parameters():

    if param.grad is not None:

        print(
            f"{name}:",
            param.grad.abs().mean().item()
        )