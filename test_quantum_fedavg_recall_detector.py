import sys
import os
import random
import math

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models.quantum_model import QuantumModel, state_circuit, n_qubits
from clients.quantum_client_recall_detector import QuantumClient
from servers.quantum_server_recall_detector import QuantumFedAvgServer



SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)

# PennyLane simulation is safest on CPU.
device = torch.device("cpu")

# Data/client settings
client_train_size = 1000
probe_size = 64              # small trusted server probe set
num_clients = 20
num_malicious = 5
rounds = 20
local_epochs = 1
batch_size = 8
learning_rate = 0.01

# Attack settings
ATTACK_TYPE = "hybrid"       # "label_flip", "delta_scale", "hybrid", "sign_flip"
ATTACK_STRENGTH = 2.0
LABEL_FLIP_PROB = 1.0

# Detection settings
# exact_top_k: select exactly num_malicious clients. Good for controlled experiments.
# recall_watchlist: select a slightly larger candidate set to catch all malicious clients,
#                  accepting more false positives.
RECALL_EXTRA_CLIENTS = 3
FIDELITY_PROBE_COUNT = 8

SAVE_PLOTS = True
SHOW_PLOTS = True

FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


def prepare_images(images):
    images = images.to(device)

    images = F.interpolate(
        images,
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
    )

    images = images.view(images.size(0), 16)
    images = images.view(images.size(0), 8, 2).mean(dim=2)

    return images


def prepare_labels(labels):
    labels = labels.to(device)
    return (labels % 2).float()


def quantum_state_fidelity(sample_input, weights_a, weights_b):
    sample_input = sample_input.detach().cpu()
    weights_a = weights_a.detach().cpu()
    weights_b = weights_b.detach().cpu()

    state_a = state_circuit(sample_input, weights_a)
    state_b = state_circuit(sample_input, weights_b)

    fidelity = torch.abs(torch.dot(torch.conj(state_a), state_b)) ** 2
    return float(fidelity.item())


def mean_server_client_fidelity(probe_inputs, global_weights, client_weights):
    
    probe_inputs = probe_inputs[:FIDELITY_PROBE_COUNT]

    fidelities = []
    for sample in probe_inputs:
        fidelities.append(
            quantum_state_fidelity(
                sample,
                global_weights,
                client_weights,
            )
        )

    return float(sum(fidelities) / len(fidelities))


def evaluate_weights_on_probe(model, weights, probe_inputs, probe_labels):
    """ Evaluate a weight vector on the trusted server probe data."""
    with torch.no_grad():
        model.weights.copy_(weights.detach().clone())
        outputs = model(probe_inputs)
        loss = F.mse_loss(outputs, probe_labels)
    return float(loss.item())


def robust_z_high(x):
    """
    Higher value = more suspicious.
    Robust z-score based on median absolute deviation.
    """
    x = x.float()
    median = x.median()
    mad = torch.median(torch.abs(x - median))

    if mad.item() < 1e-12:
        scale = x.std(unbiased=False) + 1e-12
    else:
        scale = 1.4826 * mad + 1e-12

    return (x - median) / scale


def positive_robust_z(x):
    return torch.clamp(robust_z_high(x), min=0.0)


def plot_client_metric(values, ylabel, title, filename):
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

    plt.xlabel("Client ID")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    if SAVE_PLOTS:
        plt.savefig(
            os.path.join(FIGURE_DIR, filename),
            dpi=300,
            bbox_inches="tight",
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def compute_confusion(suspicious):
    tp = fp = fn = tn = 0

    for cid in range(num_clients):
        true_is_malicious = cid < num_malicious
        pred_is_malicious = bool(suspicious[cid].item())

        if pred_is_malicious and true_is_malicious:
            tp += 1
        elif pred_is_malicious and not true_is_malicious:
            fp += 1
        elif (not pred_is_malicious) and true_is_malicious:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    accuracy = (tp + tn) / num_clients

    return tp, fp, fn, tn, precision, recall, f1, accuracy


def print_detection_report(
    title,
    scores,
    k,
    cumulative_flip_advantage,
    cumulative_probe_degradation,
    cumulative_fidelity_distance,
    cumulative_update_norm,
):
    suspicious = torch.zeros(num_clients, dtype=torch.bool)
    k = min(max(int(k), 0), num_clients)

    if k > 0:
        top_indices = torch.topk(scores, k=k).indices
        suspicious[top_indices] = True

    print(f"\n{title}")
    print(f"Detection k: {k}")
    print(
        "Client | True | CumScore | FlipAdv | ProbeDeg | FidDist | UpdNorm | Suspicious"
    )

    for cid in range(num_clients):
        true_label = "M" if cid < num_malicious else "B"
        print(
            f"{cid:02d}     | {true_label}    | "
            f"{scores[cid].item():.4f}   | "
            f"{cumulative_flip_advantage[cid].item():.4f} | "
            f"{cumulative_probe_degradation[cid].item():.4f} | "
            f"{cumulative_fidelity_distance[cid].item():.6f} | "
            f"{cumulative_update_norm[cid].item():.4f} | "
            f"{bool(suspicious[cid].item())}"
        )

    tp, fp, fn, tn, precision, recall, f1, accuracy = compute_confusion(suspicious)

    print("\nDetection Summary")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TN: {tn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    detected_ids = [cid for cid in range(num_clients) if suspicious[cid].item()]
    print("Detected suspicious clients:", detected_ids)

    return suspicious

transform = transforms.ToTensor()

full_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

generator = torch.Generator().manual_seed(SEED)

used_dataset, _ = random_split(
    full_dataset,
    [
        client_train_size + probe_size,
        len(full_dataset) - client_train_size - probe_size,
    ],
    generator=generator,
)

client_dataset, probe_dataset = random_split(
    used_dataset,
    [client_train_size, probe_size],
    generator=generator,
)

client_size = client_train_size // num_clients
split_sizes = [client_size] * num_clients
split_sizes[-1] += client_train_size - sum(split_sizes)

client_datasets = random_split(
    client_dataset,
    split_sizes,
    generator=generator,
)

probe_loader = DataLoader(
    probe_dataset,
    batch_size=probe_size,
    shuffle=False,
)

probe_images, probe_raw_labels = next(iter(probe_loader))
probe_inputs = prepare_images(probe_images)
probe_labels = prepare_labels(probe_raw_labels)
probe_flipped_labels = 1.0 - probe_labels



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
server = QuantumFedAvgServer(global_model)
probe_eval_model = QuantumModel()


weight_norms = []
weight_history = []
global_fidelities = []

client_fidelity_history = []
client_update_norm_history = []
client_probe_loss_history = []
client_flip_advantage_history = []
client_probe_degradation_history = []
client_round_score_history = []

cumulative_score = torch.zeros(num_clients)
cumulative_flip_advantage = torch.zeros(num_clients)
cumulative_probe_degradation = torch.zeros(num_clients)
cumulative_fidelity_distance = torch.zeros(num_clients)
cumulative_update_norm = torch.zeros(num_clients)

last_round_fidelities = None
last_round_update_norms = None
last_round_probe_losses = None
last_round_flip_advantages = None
last_round_probe_degradations = None
last_round_scores = None

print("\nExperiment Configuration")
print(f"Clients: {num_clients}")
print(f"Malicious clients: {num_malicious}")
print(f"Attack type: {ATTACK_TYPE}")
print(f"Attack strength: {ATTACK_STRENGTH}")
print(f"Label flip probability: {LABEL_FLIP_PROB}")
print(f"Probe samples: {probe_size}")

for round_id in range(rounds):
    global_weights = global_model.weights.detach().clone()

    global_clean_loss = evaluate_weights_on_probe(
        probe_eval_model,
        global_weights,
        probe_inputs,
        probe_labels,
    )

    client_updates = []
    round_update_norms = []
    round_fidelities = []
    round_probe_losses = []
    round_flipped_probe_losses = []
    round_flip_advantages = []
    round_probe_degradations = []

    print(f"\n--- Round {round_id + 1} ---")

    for cid, client in enumerate(clients):
        updated_weights = client.train(
            global_weights,
            epochs=local_epochs,
            lr=learning_rate,
        )

        update_norm = torch.norm(updated_weights - global_weights).item()

        avg_fidelity = mean_server_client_fidelity(
            probe_inputs,
            global_weights,
            updated_weights,
        )

        clean_loss = evaluate_weights_on_probe(
            probe_eval_model,
            updated_weights,
            probe_inputs,
            probe_labels,
        )

        flipped_loss = evaluate_weights_on_probe(
            probe_eval_model,
            updated_weights,
            probe_inputs,
            probe_flipped_labels,
        )

        # Positive value means the client update fits flipped labels better
        # than clean labels. This is very useful for the current binary label-flip attack.
        flip_advantage = clean_loss - flipped_loss

        # Positive value means the submitted client model is worse on clean
        # server probe data than the current global model.
        probe_degradation = clean_loss - global_clean_loss

        client_updates.append(updated_weights)
        round_update_norms.append(update_norm)
        round_fidelities.append(avg_fidelity)
        round_probe_losses.append(clean_loss)
        round_flipped_probe_losses.append(flipped_loss)
        round_flip_advantages.append(flip_advantage)
        round_probe_degradations.append(probe_degradation)

        tag = "M" if cid < num_malicious else "B"
        print(
            f"Client {cid:02d} [{tag}] | "
            f"UpdNorm: {update_norm:.6e} | "
            f"Fidelity: {avg_fidelity:.10f} | "
            f"ProbeLoss: {clean_loss:.6f} | "
            f"FlipAdv: {flip_advantage:.6f} | "
            f"ProbeDeg: {probe_degradation:.6f}"
        )

    fid_distance = 1.0 - torch.tensor(round_fidelities)
    update_norm_tensor = torch.tensor(round_update_norms)
    flip_adv_tensor = torch.tensor(round_flip_advantages)
    probe_deg_tensor = torch.tensor(round_probe_degradations)

    flip_score = positive_robust_z(flip_adv_tensor)
    probe_deg_score = positive_robust_z(probe_deg_tensor)
    fid_score = positive_robust_z(fid_distance)
    norm_score = positive_robust_z(update_norm_tensor)

    # For a binary/hybrid label-flip attack, the clean-vs-flipped probe signal
    # is the strongest indicator. Fidelity and update norm remain supportive signals.
    round_score = (
        0.50 * flip_score
        + 0.20 * probe_deg_score
        + 0.15 * fid_score
        + 0.15 * norm_score
    )

    cumulative_score += round_score.detach().cpu()
    cumulative_flip_advantage += flip_adv_tensor.detach().cpu()
    cumulative_probe_degradation += probe_deg_tensor.detach().cpu()
    cumulative_fidelity_distance += fid_distance.detach().cpu()
    cumulative_update_norm += update_norm_tensor.detach().cpu()

    client_fidelity_history.append(round_fidelities)
    client_update_norm_history.append(round_update_norms)
    client_probe_loss_history.append(round_probe_losses)
    client_flip_advantage_history.append(round_flip_advantages)
    client_probe_degradation_history.append(round_probe_degradations)
    client_round_score_history.append(round_score.detach().cpu().tolist())

    last_round_fidelities = round_fidelities
    last_round_update_norms = round_update_norms
    last_round_probe_losses = round_probe_losses
    last_round_flip_advantages = round_flip_advantages
    last_round_probe_degradations = round_probe_degradations
    last_round_scores = round_score.detach().cpu().tolist()

    manual_avg = torch.stack(client_updates).mean(dim=0)
    manual_server_delta = torch.norm(manual_avg - global_weights).item()

    # Keep normal FedAvg aggregation for evaluation. The detector reports suspicious
    # clients, but does not filter them here. Filtering can be added after validation.
    server.aggregate(client_updates)

    actual_server_delta = torch.norm(global_model.weights.detach() - global_weights).item()

    weight_history.append(global_model.weights.detach().clone())

    norm = torch.norm(global_model.weights).item()
    weight_norms.append(norm)

    print(
        f"Round {round_id + 1} | "
        f"Weight Norm: {norm:.10f} | "
        f"Mean Client Update: {sum(round_update_norms) / len(round_update_norms):.10e} | "
        f"Manual Server Delta: {manual_server_delta:.10e} | "
        f"Actual Server Delta: {actual_server_delta:.10e}"
    )


# ============================================================
# Global round-to-round fidelity
# ============================================================

for i in range(len(weight_history) - 1):
    fidelity = mean_server_client_fidelity(
        probe_inputs,
        weight_history[i],
        weight_history[i + 1],
    )
    global_fidelities.append(fidelity)

print("\nGlobal Round-to-Round Fidelity Values:\n")
for i, fidelity in enumerate(global_fidelities):
    print(f"Round {i + 1} -> {i + 2}: {fidelity:.6f}")


# ============================================================
# Detection reports
# ============================================================

print_detection_report(
    title="\nMulti-Round Detector: Exact Top-k Result",
    scores=cumulative_score,
    k=num_malicious,
    cumulative_flip_advantage=cumulative_flip_advantage,
    cumulative_probe_degradation=cumulative_probe_degradation,
    cumulative_fidelity_distance=cumulative_fidelity_distance,
    cumulative_update_norm=cumulative_update_norm,
)

recall_k = min(num_clients, num_malicious + RECALL_EXTRA_CLIENTS)
print_detection_report(
    title="\nMulti-Round Detector: Recall-First Watchlist Result",
    scores=cumulative_score,
    k=recall_k,
    cumulative_flip_advantage=cumulative_flip_advantage,
    cumulative_probe_degradation=cumulative_probe_degradation,
    cumulative_fidelity_distance=cumulative_fidelity_distance,
    cumulative_update_norm=cumulative_update_norm,
)


# ============================================================
# Plots
# ============================================================

plt.figure()
plt.plot(weight_norms)
plt.xlabel("Communication Round")
plt.ylabel("Global Weight Norm")
plt.title("Quantum FedAvg Convergence")
plt.grid()

if SAVE_PLOTS:
    plt.savefig(os.path.join(FIGURE_DIR, "qfl_recall_detector_convergence.png"), dpi=300, bbox_inches="tight")
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
    plt.savefig(os.path.join(FIGURE_DIR, "qfl_recall_detector_global_fidelity.png"), dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

plot_client_metric(
    last_round_fidelities,
    "Server-Client Fidelity",
    "Last-Round Client Fidelity",
    "qfl_recall_detector_last_round_fidelity.png",
)

plot_client_metric(
    last_round_update_norms,
    "Update Norm",
    "Last-Round Client Update Norms",
    "qfl_recall_detector_last_round_update_norms.png",
)

plot_client_metric(
    last_round_flip_advantages,
    "Clean Loss - Flipped Loss",
    "Last-Round Label-Flip Advantage",
    "qfl_recall_detector_last_round_flip_advantage.png",
)

plot_client_metric(
    cumulative_score.tolist(),
    "Cumulative Detection Score",
    "Multi-Round Detection Score",
    "qfl_recall_detector_cumulative_scores.png",
)

plot_client_metric(
    cumulative_flip_advantage.tolist(),
    "Cumulative Flip Advantage",
    "Multi-Round Label-Flip Evidence",
    "qfl_recall_detector_cumulative_flip_advantage.png",
)

print("\nDone. Plots saved in:", FIGURE_DIR)
