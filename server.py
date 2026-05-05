import numpy as np

class Server:
    def __init__(self, clients):
        self.clients = clients

    def aggregate(self, weights_list):
        return np.mean(weights_list, axis=0)

    def train(self, rounds=5):
        global_weights = np.random.randn(4)

        for r in range(rounds):
            client_updates = []

            for client in self.clients:
                updated = client.train(global_weights)
                client_updates.append(updated)

            global_weights = self.aggregate(client_updates)

            print(f"Round {r+1} completed")

        return global_weights