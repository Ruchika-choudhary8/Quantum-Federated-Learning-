import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

from models.classical_model import ClassicalCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Tiny dataset (10 samples)
tiny_subset = Subset(dataset, list(range(10)))

loader = DataLoader(tiny_subset, batch_size=10, shuffle=True)

model = ClassicalCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(50):

    model.train()

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    # Accuracy
    with torch.no_grad():

        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()

    print(
        f"Epoch {epoch+1} | "
        f"Loss: {loss.item():.6f} | "
        f"Accuracy: {acc.item()*100:.2f}%"
    )