import torch
import torch.nn as nn
import pennylane as qml


# Number of qubits
n_qubits = 4


# Quantum device
dev = qml.device(
    "default.qubit",
    wires=n_qubits
)


# Quantum Circuit
@qml.qnode(
    dev,
    interface="torch"
)
def circuit(inputs, weights):

    # -----------------------------
    # Data Encoding
    # -----------------------------
    for i in range(n_qubits):

        qml.RX(
            inputs[i],
            wires=i
        )

    # -----------------------------
    # Trainable Quantum Layer
    # -----------------------------
    for i in range(n_qubits):

        qml.RY(
            weights[i],
            wires=i
        )

    # -----------------------------
    # Entanglement Layer
    # -----------------------------
    for i in range(n_qubits - 1):

        qml.CNOT(
            wires=[i, i + 1]
        )

    # -----------------------------
    # Multi-Qubit Measurement
    # -----------------------------
    return [
        qml.expval(
            qml.PauliZ(i)
        )
        for i in range(n_qubits)
    ]


# Quantum Neural Network
class QuantumModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Trainable quantum parameters
        self.weights = nn.Parameter(

            0.01 * torch.randn(
                n_qubits
            )
        )

    def forward(self, x):

        outputs = []

        for sample in x:

            # Quantum circuit output
            out = circuit(
                sample,
                self.weights
            )

            # Convert list to tensor
            out = torch.stack(out)

            # Aggregate measurements
            out = out.mean()

            outputs.append(out)

        outputs = torch.stack(outputs)

        return outputs