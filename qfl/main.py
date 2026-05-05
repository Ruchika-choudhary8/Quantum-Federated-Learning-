import numpy as np
from mnist_loader import load_mnist
from client import Client
from server import Server
from model import quantum_model

# Load MNIST
X_train, y_train = load_mnist(
    "mnist-dataset/versions/1/train-images.idx3-ubyte",
    "mnist-dataset/versions/1/train-labels.idx1-ubyte"
)

X_test, y_test = load_mnist(
    "mnist-dataset/versions/1/t10k-images.idx3-ubyte",
    "mnist-dataset/versions/1/t10k-labels.idx1-ubyte"
)

# 🔹 Reduce to binary classification (0 vs 1)
mask = (y_train == 0) | (y_train == 1)
X_train, y_train = X_train[mask], y_train[mask]

mask = (y_test == 0) | (y_test == 1)
X_test, y_test = X_test[mask], y_test[mask]

# 🔹 Reduce features (quantum limitation)
X_train = X_train[:, :4]
X_test = X_test[:, :4]

# Create clients
def create_clients(X, y, num_clients=3):
    clients = []
    size = len(X) // num_clients

    for i in range(num_clients):
        X_i = X[i*size:(i+1)*size]
        y_i = y[i*size:(i+1)*size]
        clients.append(Client(X_i, y_i))

    return clients

clients = create_clients(X_train, y_train)

# Train QFL
server = Server(clients)
global_weights = server.train(rounds=5)

# 🔹 Evaluate
correct = 0

for x, y in zip(X_test, y_test):
    pred = quantum_model(global_weights, x)
    pred_label = 1 if pred > 0 else 0

    if pred_label == y:
        correct += 1

accuracy = correct / len(X_test)

print("\nQFL Accuracy:", accuracy)