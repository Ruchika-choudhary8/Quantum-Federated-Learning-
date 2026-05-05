import numpy as np
from mnist_loader import load_mnist
from client import Client
from server import Server
from model import Model

X_train, y_train = load_mnist(
    "../mnist-dataset/versions/1/train-images.idx3-ubyte",
    "../mnist-dataset/versions/1/train-labels.idx1-ubyte"
)

X_test, y_test = load_mnist(
    "../mnist-dataset/versions/1/t10k-images.idx3-ubyte",
    "../mnist-dataset/versions/1/t10k-labels.idx1-ubyte"
)

def create_clients(X, y, num_clients=5):
    clients = []
    size = len(X) // num_clients

    for i in range(num_clients):
        X_i = X[i*size:(i+1)*size]
        y_i = y[i*size:(i+1)*size]
        clients.append(Client(X_i, y_i))

    return clients

clients = create_clients(X_train, y_train, num_clients=5)

server = Server(clients)
global_weights = server.train(rounds=5)

model = Model()
model.weights = global_weights

preds = model.predict(X_test)
accuracy = np.mean(preds == y_test)

print("\nFinal Accuracy:", accuracy)