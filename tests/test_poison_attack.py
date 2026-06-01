import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.classical_model import ClassicalCNN
from client import ClassicalClient
from server import FedAvgServer
from utils.evaluate import evaluate
from utils.data_split import split_dataset


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# MNIST
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Split clients
num_clients = 5

client_datasets = split_dataset(
    train_dataset,
    num_clients
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Global model
global_model = ClassicalCNN().to(device)

clients = []
#malicious client
malicious_client = [0,1]

for i, dataset in enumerate(client_datasets):

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    client = ClassicalClient(
        ClassicalCNN().to(device),
        loader,
        device,
        malicious=(i in malicious_client)
    )

    clients.append(client)

server = FedAvgServer(global_model)

rounds = 20

accuracies = []

for r in range(rounds):

    client_states = []

    global_state = global_model.state_dict()

    for client in clients:

        updated_state = client.train(
            global_state,
            epochs=3
        )

        client_states.append(updated_state)

    server.aggregate(client_states)

    acc = evaluate(
        global_model,
        test_loader,
        device
    )

    accuracies.append(acc)

    print(
        f"Round {r+1} | "
        f"Accuracy: {acc:.2f}%"
    )

# Plot
plt.plot(accuracies)

plt.xlabel("Communication Round")
plt.ylabel("Accuracy")

plt.title("FedAvg Under Poison Attack")

plt.grid()

plt.show()