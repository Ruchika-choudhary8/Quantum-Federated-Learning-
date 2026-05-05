import pennylane as qml
import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_model(weights, x):
    # Encode first 4 features
    for i in range(n_qubits):
        qml.RX(x[i], wires=i) # To convert classical data to quantum state

  
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i) # weights = theta(parameters)

    return qml.expval(qml.PauliZ(0))  # binary output(measurement)