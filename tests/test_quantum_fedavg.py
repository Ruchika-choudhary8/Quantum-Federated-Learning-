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
import matplotlib.pyplot as plt

from torchvision import (
    datasets,
    transforms
)

from torch.utils.data import (
    DataLoader,
    random_split
)

from models.quantum_model import QuantumModel

from clients.quantum_client import QuantumClient

from servers.quantum_server import (
    QuantumFedAvgServer
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

subset_size = 1000

dataset, _ = random_split(

    dataset,

    [
        subset_size,
        len(dataset) - subset_size
    ]
)

num_clients = 5

client_size = (
    subset_size // num_clients
)

client_datasets = random_split(

    dataset,

    [client_size] * num_clients
)

clients = []

for data in client_datasets:

    loader = DataLoader(
        data,
        batch_size=16,
        shuffle=True
    )

    client = QuantumClient(

        QuantumModel(),

        loader,

        device
    )

    clients.append(client)

global_model = QuantumModel()

server = QuantumFedAvgServer(
    global_model
)

rounds = 20

weight_norms = []
weight_history = []

for r in range(rounds):

    global_weights = (
        global_model.weights
        .detach()
        .clone()
    )

    client_updates = []

    for client in clients:

        updated_weights = client.train(

            global_weights,

            epochs=1
        )

        client_updates.append(
            updated_weights
        )

    server.aggregate(
        client_updates
    )
    
    weight_history.append(
        global_model.weights.detach().clone()
    )

    norm = torch.norm(
        global_model.weights
    ).item()

    weight_norms.append(
        norm
    )

    print(
        f"Round {r+1} | "
        f"Weight Norm: "
        f"{norm:.4f}"
    )

plt.plot(weight_norms)

plt.xlabel(
    "Communication Round"
)

plt.ylabel(
    "Global Weight Norm"
)

plt.title(
    "Quantum FedAvg Convergence"
)

plt.grid()

plt.show()

fidelities = []

for i in range(len(weight_history) - 1):

    w1 = weight_history[i]
    w2 = weight_history[i + 1]

    fidelity = torch.dot(
        w1,
        w2
    ) / (
        torch.norm(w1)
        * torch.norm(w2)
    )

    fidelities.append(
        fidelity.item()
    )

print("\nFidelity Values:\n")

for i, f in enumerate(fidelities):

    print(
        f"Round {i+1} -> {i+2}: "
        f"{f:.6f}"
    )

plt.figure()

plt.plot(fidelities)

plt.xlabel(
    "Communication Round"
)

plt.ylabel(
    "Fidelity"
)

plt.title(
    "Quantum Model Fidelity"
)

plt.grid()

plt.show()