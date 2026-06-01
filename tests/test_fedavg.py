import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch
import matplotlib.pyplot as plt

from models.classical_model import ClassicalCNN
from servers.fedavg import fedavg_aggregate


# Create models
model_a = ClassicalCNN()
model_b = ClassicalCNN()


# Fill weights manually
for param in model_a.parameters():
    param.data.fill_(1.0)

for param in model_b.parameters():
    param.data.fill_(3.0)


# Get state dictionaries
state_a = model_a.state_dict()
state_b = model_b.state_dict()


# Perform FedAvg aggregation
global_state = fedavg_aggregate(
    [state_a, state_b],
    [1, 1]
)


# Verification

all_correct = True

error = 0

for key in global_state:

    expected = torch.full_like(
        global_state[key],
        2.0
    )

    if not torch.allclose(
        global_state[key],
        expected
    ):

        print(f"FAILED at layer: {key}")

        all_correct = False

    error += torch.sum(
        torch.abs(
            global_state[key] - expected
        )
    )


sample_key = list(global_state.keys())[0]

print("\nClient A Weights:")
print(state_a[sample_key][0][:5])

print("\nClient B Weights:")
print(state_b[sample_key][0][:5])

print("\nGlobal Aggregated Weights:")
print(global_state[sample_key][0][:5])


print("\nTotal Aggregation Error:", error.item())



weights_a = (
    state_a[sample_key]
    .flatten()
    .detach()
    .numpy()
)

weights_b = (
    state_b[sample_key]
    .flatten()
    .detach()
    .numpy()
)

weights_g = (
    global_state[sample_key]
    .flatten()
    .detach()
    .numpy()
)

plt.hist(
    weights_a,
    alpha=0.5,
    label="Client A"
)

plt.hist(
    weights_b,
    alpha=0.5,
    label="Client B"
)

plt.hist(
    weights_g,
    alpha=0.5,
    label="Global"
)

plt.xlabel("Weight Values")
plt.ylabel("Frequency")

plt.title("FedAvg Weight Aggregation Verification")

plt.legend()

plt.grid()

plt.show()


if all_correct:

    print("\nFedAvg aggregation PASSED")
else:

    print("\nFedAvg aggregation FAILED")