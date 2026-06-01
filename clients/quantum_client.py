import torch
import torch.nn as nn
import torch.optim as optim


class QuantumClient:

    def __init__(
        self,
        model,
        train_loader,
        device
    ):

        self.model = model
        self.train_loader = train_loader
        self.device = device

    def train(
        self,
        global_weights,
        epochs=1,
        lr=0.01
    ):

        self.model.weights.data = (
            global_weights.clone()
        )

        criterion = nn.MSELoss()

        optimizer = optim.Adam(
            [self.model.weights],
            lr=lr
        )

        self.model.train()

        for epoch in range(epochs):

            for images, labels in self.train_loader:

                images = images.to(self.device)

                images = images.view(
                    images.size(0),
                    -1
                )[:, :4]

                labels = (
                    labels % 2
                ).float().to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

        return (
            self.model.weights
            .detach()
            .clone()
        )