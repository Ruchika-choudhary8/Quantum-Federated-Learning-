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
import pennylane as qml

from torchvision import (
    datasets,
    transforms
)

from torch.utils.data import (
    DataLoader,
    random_split
)

from models.quantum_model import (
    QuantumModel,
    state_circuit
)

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

num_clients = 20

client_size = subset_size // num_clients

split_sizes = [client_size] * num_clients

split_sizes[-1] += (
    subset_size - sum(split_sizes)
)

client_datasets = random_split(
    dataset,
    split_sizes
)

num_malicious = 10

clients = []

for i, data in enumerate(client_datasets):

    loader = DataLoader(
        data,
        batch_size=8,
        shuffle=True
    )

    malicious = i < num_malicious


    client = QuantumClient(
        QuantumModel(),
        loader,
        device,
        malicious=malicious
    )

    clients.append(client)

global_model = QuantumModel()

server = QuantumFedAvgServer(
    global_model
)

rounds = 20

weight_norms = []
weight_history = []
client_fidelity_history = []

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
        round_fidelities = []

        sample_input = torch.tensor(
             [0.5] * 8
        )

        for client_weights in client_updates:

             state_global = state_circuit(
                  sample_input,
                  global_weights
             )

             state_client = state_circuit(
                  sample_input,
                  client_weights
             )

             dm_global = qml.math.dm_from_state_vector(
                  state_global
             )

             dm_client = qml.math.dm_from_state_vector(
                  state_client
             )

             fidelity = qml.math.fidelity(
                  dm_global,
                  dm_client
             )

             round_fidelities.append(
                  float(fidelity)
             )

        client_fidelity_history.append(
             round_fidelities
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

sample_input = torch.tensor(
    [0.5] * 8
)

for i in range(len(weight_history) - 1):

    state1 = state_circuit(
        sample_input,
        weight_history[i]
    )

    state2 = state_circuit(
        sample_input,
        weight_history[i + 1]
    )

    fidelity = torch.abs(
        torch.dot(
            torch.conj(state1),
            state2
        )
    ) ** 2
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

plt.figure(
    figsize=(8,5)
)

last_round = (
    client_fidelity_history[-1]
)

for i, fid in enumerate(last_round):

    if i < num_malicious:

        plt.scatter(
            i,
            fid,
            marker="x",
            s=120,
            label="Malicious"
            if i == 0 else ""
        )

    else:

        plt.scatter(
            i,
            fid,
            s=120,
            label="Benign"
            if i == num_malicious else ""
        )

plt.xlabel(
    "Client ID"
)

plt.ylabel(
    "Server-Client Fidelity"
)

plt.title(
    "Client Fidelity Clustering"
)

plt.legend()

plt.grid()

plt.show()
