import numpy as np

class Server:
    def __init__(self, clients):
        self.clients = clients

    def aggregate(self, weights_list):
        return np.mean(weights_list, axis=0)

    def train(self, rounds=5):
        input_size = 784
        num_classes = 10

        global_weights = np.zeros((input_size, num_classes))

        for r in range(rounds):
            local_weights = []

            for client in self.clients:
                updated = client.train(global_weights)
                local_weights.append(updated)

            global_weights = self.aggregate(local_weights)

            print(f"Round {r+1} completed")

        return global_weights