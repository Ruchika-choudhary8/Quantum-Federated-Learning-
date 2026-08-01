import sys
import os
import random

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import matplotlib.pyplot as plt

from torchvision import (
    datasets,
    transforms,
)

from torch.utils.data import (
    DataLoader,
    random_split,
)

from models.quantum_model import (
    QuantumModel,
    state_circuit,
    n_qubits,
)

from clients.quantum_client_detection_ready import QuantumClient

from servers.quantum_server_detection_ready import QuantumFedAvgServer


SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)

# PennyLane statevector simulation is CPU-oriented.
device = torch.device("cpu")

subset_size = 1000
num_clients = 20
num_malicious = 5
rounds = 20
local_epochs = 1
batch_size = 8
learning_rate = 0.01

# Attack configuration
ATTACK_TYPE = "hybrid"
ATTACK_STRENGTH = 2.0
LABEL_FLIP_PROB = 1.0

# Fidelity-only detection configuration
# "top_k" is best for controlled experiments where num_malicious is known.
# "mad_threshold" is better for a more realistic setting where num_malicious is unknown.
FIDELITY_DETECTION_MODE = "top_k"
FIDELITY_MAD_FACTOR = 1.5

SAVE_PLOTS = True
SHOW_PLOTS = True

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "figures",
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)


# -----------------------------
# Helper functions
# -----------------------------


def quantum_state_fidelity(
    sample_input,
    weights_a,
    weights_b,
):
    """
    Pure-state fidelity between two quantum circuits with different weights.
    Lower server-client fidelity means the client update moved farther from
    the current global quantum model.
    """
    sample_input = sample_input.detach().cpu()
    weights_a = weights_a.detach().cpu()
    weights_b = weights_b.detach().cpu()

    state_a = state_circuit(
        sample_input,
        weights_a,
    )

    state_b = state_circuit(
        sample_input,
        weights_b,
    )

    fidelity = torch.abs(
        torch.dot(
            torch.conj(state_a),
            state_b,
        )
    ) ** 2

    return float(fidelity.item())


def plot_client_metric(
    values,
    ylabel,
    title,
    filename,
    horizontal_line=None,
    horizontal_label=None,
):
    plt.figure(figsize=(8, 5))

    for cid, value in enumerate(values):
        if cid < num_malicious:
            plt.scatter(
                cid,
                value,
                marker="x",
                s=120,
                label="Malicious" if cid == 0 else "",
            )
        else:
            plt.scatter(
                cid,
                value,
                s=120,
                label="Benign" if cid == num_malicious else "",
            )

    if horizontal_line is not None:
        plt.axhline(
            y=horizontal_line,
            linestyle="--",
            linewidth=1.5,
            label=horizontal_label,
        )

    plt.xlabel("Client ID")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    if SAVE_PLOTS:
        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                filename,
            ),
            dpi=300,
            bbox_inches="tight",
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def fidelity_only_detection_summary(
    last_round_fidelities,
):
    """
    Detect malicious clients using only server-client quantum fidelity.

    Controlled top-k rule:
        Select the k clients with the lowest fidelity, where k=num_malicious.

    MAD threshold rule:
        Flag clients with fidelity below median - factor * MAD-scale.
    """
    fid_tensor = torch.tensor(
        last_round_fidelities,
        dtype=torch.float32,
    )

    fidelity_distance = 1.0 - fid_tensor

    suspicious = torch.zeros(
        num_clients,
        dtype=torch.bool,
    )

    threshold = None

    if FIDELITY_DETECTION_MODE == "top_k":
        if num_malicious > 0:
            k = min(
                num_malicious,
                num_clients,
            )

            # Largest fidelity distance = lowest fidelity.
            top_k_indices = torch.topk(
                fidelity_distance,
                k=k,
            ).indices

            suspicious[top_k_indices] = True

            # Equivalent fidelity threshold for plotting/reporting.
            threshold = fid_tensor[top_k_indices].max().item()

    elif FIDELITY_DETECTION_MODE == "mad_threshold":
        median_fid = fid_tensor.median()
        mad = torch.median(
            torch.abs(
                fid_tensor - median_fid,
            )
        )

        scale = 1.4826 * mad + 1e-12
        threshold = (
            median_fid - FIDELITY_MAD_FACTOR * scale
        ).item()

        suspicious = fid_tensor < threshold

    else:
        raise ValueError(
            "FIDELITY_DETECTION_MODE must be 'top_k' or 'mad_threshold'."
        )

    print("\nFidelity-Only Detection Table")
    print(
        "Client | True | Fidelity    | Fidelity Distance | Suspicious"
    )

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for cid in range(num_clients):
        true_is_malicious = cid < num_malicious
        pred_is_malicious = bool(
            suspicious[cid].item()
        )

        if pred_is_malicious and true_is_malicious:
            true_positive += 1
        elif pred_is_malicious and not true_is_malicious:
            false_positive += 1
        elif (not pred_is_malicious) and true_is_malicious:
            false_negative += 1
        else:
            true_negative += 1

        true_label = "M" if true_is_malicious else "B"

        print(
            f"{cid:02d}     | {true_label}    | "
            f"{fid_tensor[cid].item():.10f} | "
            f"{fidelity_distance[cid].item():.10e} | "
            f"{pred_is_malicious}"
        )

    precision = true_positive / (
        true_positive + false_positive + 1e-12
    )

    recall = true_positive / (
        true_positive + false_negative + 1e-12
    )

    f1 = 2 * precision * recall / (
        precision + recall + 1e-12
    )

    accuracy = (
        true_positive + true_negative
    ) / num_clients

    print("\nFidelity-Only Detection Summary")
    print(f"Detection mode: {FIDELITY_DETECTION_MODE}")
    if threshold is not None:
        print(f"Fidelity threshold: {threshold:.10f}")
    print(f"TP: {true_positive}")
    print(f"FP: {false_positive}")
    print(f"FN: {false_negative}")
    print(f"TN: {true_negative}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    plot_client_metric(
        fidelity_distance.tolist(),
        "Fidelity Distance (1 - Fidelity)",
        "Fidelity-Only Suspicion Score",
        "qfl_fidelity_only_suspicion_score.png",
    )

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "threshold": threshold,
    }


# -----------------------------
# Dataset preparation
# -----------------------------

transform = transforms.ToTensor()

full_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

generator = torch.Generator().manual_seed(SEED)

dataset, _ = random_split(
    full_dataset,
    [
        subset_size,
        len(full_dataset) - subset_size,
    ],
    generator=generator,
)

client_size = subset_size // num_clients
split_sizes = [client_size] * num_clients
split_sizes[-1] += subset_size - sum(split_sizes)

client_datasets = random_split(
    dataset,
    split_sizes,
    generator=generator,
)


# -----------------------------
# Client and server setup
# -----------------------------

clients = []

for cid, data in enumerate(client_datasets):
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
    )

    malicious = cid < num_malicious

    client = QuantumClient(
        QuantumModel(),
        loader,
        device,
        malicious=malicious,
        attack_type=ATTACK_TYPE if malicious else "none",
        attack_strength=ATTACK_STRENGTH,
        label_flip_prob=LABEL_FLIP_PROB,
    )

    clients.append(client)

global_model = QuantumModel()

server = QuantumFedAvgServer(
    global_model,
)


# -----------------------------
# Federated training
# -----------------------------

sample_input = torch.tensor(
    [0.5] * n_qubits,
    dtype=torch.float32,
)

weight_norms = []
weight_history = []
global_fidelities = []
client_fidelity_history = []
client_update_norm_history = []

last_round_fidelities = None
last_round_update_norms = None

print("\nExperiment Configuration")
print(f"Clients: {num_clients}")
print(f"Malicious clients: {num_malicious}")
print(f"Attack type: {ATTACK_TYPE}")
print(f"Attack strength: {ATTACK_STRENGTH}")
print(f"Label flip probability: {LABEL_FLIP_PROB}")
print(f"Fidelity-only detection mode: {FIDELITY_DETECTION_MODE}")

for round_id in range(rounds):
    global_weights = global_model.weights.detach().clone()

    client_updates = []
    round_update_norms = []
    round_fidelities = []

    print(f"\n--- Round {round_id + 1} ---")

    for cid, client in enumerate(clients):
        updated_weights = client.train(
            global_weights,
            epochs=local_epochs,
            lr=learning_rate,
        )

        update_norm = torch.norm(
            updated_weights - global_weights,
        ).item()

        server_client_fidelity = quantum_state_fidelity(
            sample_input,
            global_weights,
            updated_weights,
        )

        client_updates.append(updated_weights)
        round_update_norms.append(update_norm)
        round_fidelities.append(server_client_fidelity)

        tag = "M" if cid < num_malicious else "B"

        print(
            f"Client {cid:02d} [{tag}] | "
            f"Update Norm: {update_norm:.10e} | "
            f"Server-Client Fidelity: {server_client_fidelity:.10f}"
        )

    manual_avg = torch.stack(
        client_updates,
    ).mean(dim=0)

    manual_server_delta = torch.norm(
        manual_avg - global_weights,
    ).item()

    server.aggregate(client_updates)

    actual_server_delta = torch.norm(
        global_model.weights.detach() - global_weights,
    ).item()

    weight_history.append(
        global_model.weights.detach().clone(),
    )

    norm = torch.norm(
        global_model.weights,
    ).item()

    weight_norms.append(norm)
    client_fidelity_history.append(round_fidelities)
    client_update_norm_history.append(round_update_norms)

    last_round_fidelities = round_fidelities
    last_round_update_norms = round_update_norms

    print(
        f"Round {round_id + 1} | "
        f"Weight Norm: {norm:.10f} | "
        f"Mean Client Update: {sum(round_update_norms) / len(round_update_norms):.10e} | "
        f"Manual Server Delta: {manual_server_delta:.10e} | "
        f"Actual Server Delta: {actual_server_delta:.10e}"
    )


# -----------------------------
# Global round-to-round fidelity
# -----------------------------

for i in range(len(weight_history) - 1):
    fidelity = quantum_state_fidelity(
        sample_input,
        weight_history[i],
        weight_history[i + 1],
    )

    global_fidelities.append(fidelity)

print("\nGlobal Round-to-Round Fidelity Values:\n")

for i, fidelity in enumerate(global_fidelities):
    print(
        f"Round {i + 1} -> {i + 2}: "
        f"{fidelity:.6f}"
    )


# -----------------------------
# Plots
# -----------------------------

plt.figure()
plt.plot(weight_norms)
plt.xlabel("Communication Round")
plt.ylabel("Global Weight Norm")
plt.title("Quantum FedAvg Convergence")
plt.grid()

if SAVE_PLOTS:
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "qfl_fedavg_convergence_fidelity_only.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

plt.figure()
plt.plot(global_fidelities)
plt.xlabel("Communication Round")
plt.ylabel("Fidelity")
plt.title("Quantum Model Fidelity")
plt.grid()

if SAVE_PLOTS:
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "qfl_global_model_fidelity_fidelity_only.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

plot_client_metric(
    last_round_update_norms,
    "Update Norm",
    "Client Update Norms",
    "qfl_client_update_norms_fidelity_only.png",
)

# Threshold will be calculated inside the detection summary, but we also plot raw fidelity here.
plot_client_metric(
    last_round_fidelities,
    "Server-Client Fidelity",
    "Client Fidelity Clustering",
    "qfl_client_fidelity_clustering_fidelity_only.png",
)


# -----------------------------
# Fidelity-only malicious-client detection
# -----------------------------

fidelity_only_detection_summary(
    last_round_fidelities,
)
