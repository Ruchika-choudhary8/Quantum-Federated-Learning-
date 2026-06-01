import torch


class QuantumFedAvgServer:

    def __init__(
        self,
        global_model
    ):

        self.global_model = global_model

    def aggregate(
        self,
        client_weights
    ):

        avg_weights = torch.mean(

            torch.stack(
                client_weights
            ),

            dim=0
        )

        self.global_model.weights.data = (
            avg_weights.clone()
        )

        return avg_weights