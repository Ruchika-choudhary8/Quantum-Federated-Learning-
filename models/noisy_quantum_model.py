import torch
import torch.nn as nn
import pennylane as qml


n_qubits = 4

dev = qml.device(
    "default.mixed",
    wires=n_qubits
)

@qml.qnode(
    dev,
    interface="torch"
)
def noisy_circuit(
    inputs,
    weights,
    p
):

    for i in range(n_qubits):

        qml.RX(
            inputs[i],
            wires=i
        )

        qml.DepolarizingChannel(
            p,
            wires=i
        )

    for i in range(n_qubits):

        qml.RY(
            weights[i],
            wires=i
        )

        qml.DepolarizingChannel(
            p,
            wires=i
        )

    for i in range(n_qubits - 1):

        qml.CNOT(
            wires=[i, i + 1]
        )

    return [
        qml.expval(
            qml.PauliZ(i)
        )
        for i in range(n_qubits)
    ]


class NoisyQuantumModel(nn.Module):

    def __init__(
        self,
        p=0.0
    ):

        super().__init__()

        self.p = p

        self.weights = nn.Parameter(
            0.01 * torch.randn(n_qubits)
        )

    def forward(self, x):

        outputs = []

        for sample in x:

            out = noisy_circuit(
                sample,
                self.weights,
                self.p
            )

            out = torch.stack(out)

            out = out.mean()

            outputs.append(out)

        return torch.stack(outputs)