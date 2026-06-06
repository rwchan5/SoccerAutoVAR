"""
Training script for the foul severity classifier.

Run with:
    conda run -n SoccerNet python train.py --group inside
    conda run -n SoccerNet python train.py --group outside
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv

from dataset import FoulDataset
from model import build_model, count_params

DATA_DIR   = Path(__file__).parent / "data"
VIDEO_ROOT = Path(__file__).parent.parent / "SoccerNetData" / "mvfouls"
CKPT_DIR   = Path(__file__).parent / "checkpoints"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE    = 4
NUM_EPOCHS    = 40
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10   # early stopping: halt if val acc doesn't improve for this many epochs
NUM_WORKERS   = 0    # MPS + multiprocessing deadlocks on Mac
# ──────────────────────────────────────────────────────────────────────────────


def compute_class_weights(csv_path: Path, binary: bool, num_classes: int, device) -> torch.Tensor:
    """
    Compute inverse-frequency weights per class so the loss penalises mistakes
    on minority classes (e.g. no-foul) proportionally more.
    """
    from dataset import SEVERITY_TO_BINARY, SEVERITY_TO_CLASS
    label_map = SEVERITY_TO_BINARY if binary else SEVERITY_TO_CLASS
    counts = [0] * num_classes
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            counts[label_map[row["severity"]]] += 1
    total = sum(counts)
    # weight = total / (num_classes * count) — rare classes get higher weight
    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32).to(device)


def make_loaders(group: str, binary: bool):
    train_ds = FoulDataset(DATA_DIR / f"{group}_train.csv", VIDEO_ROOT, train=True,  binary=binary)
    val_ds   = FoulDataset(DATA_DIR / f"{group}_val.csv",   VIDEO_ROOT, train=False, binary=binary)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for frames, labels, _ in loader:   # incident_idx not needed during training
        frames = frames.to(device)
        labels = labels.to(device)

        # TODO: zero the gradients
        optimizer.zero_grad()
        # TODO: forward pass — run frames through model to get logits
        logits = model(frames)
        # TODO: compute loss between logits and labels
        loss = loss_fn(logits, labels)
        # TODO: backward pass
        loss.backward()
        # TODO: optimizer step
        optimizer.step()
        # TODO: accumulate total_loss, correct, total
        #       hint: predictions = logits.argmax(dim=1)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()


    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, loss_fn, device):
    """
    Multi-clip inference: run every (incident, clip) pair through the model,
    average the softmax scores per incident, then take argmax for the prediction.
    """
    model.eval()

    # Accumulate per-incident: softmax scores and the true label
    incident_scores = defaultdict(list)   # incident_idx -> list of softmax vectors
    incident_labels = {}                  # incident_idx -> true label (same for all clips)

    with torch.no_grad():
        for frames, labels, incident_idxs in loader:
            frames = frames.to(device)

            # TODO: forward pass to get logits
            logits = model(frames)
            # TODO: convert logits to probabilities with softmax (dim=1)
            probs = torch.softmax(logits, dim=1)
            # TODO: for each item in the batch, append its softmax vector to
            #       incident_scores[incident_idx] and record its label in
            #       incident_labels[incident_idx]
            #       hint: iterate over zip(incident_idxs, probs, labels)
            for incident_idx, prob, label in zip(incident_idxs, probs, labels):
                incident_scores[incident_idx].append(prob)
                incident_labels[incident_idx] = label
            pass

    # Aggregate: average clips, compute loss + accuracy across incidents
    total_loss = 0.0
    correct    = 0

    for inc_idx, scores in incident_scores.items():
        # TODO: stack the list of softmax vectors and average them
        #       avg_probs shape: (6,)
        avg_probs = torch.stack(scores).mean(dim=0)
        # TODO: predicted class = argmax of avg_probs
        predicted_class = avg_probs.argmax()
        # TODO: true label = incident_labels[inc_idx]
        true_label = incident_labels[inc_idx]
        # TODO: check if prediction == true label, accumulate correct
        if predicted_class == true_label:
            correct += 1
        # TODO: compute cross-entropy loss from avg_probs
        #       hint: loss_fn expects logits, but you can use
        #             -torch.log(avg_probs[true_label]) as a simple NLL loss here
        loss = -torch.log(avg_probs[true_label])
        total_loss += loss.item()

    n = len(incident_scores)
    avg_loss = total_loss / n
    accuracy = correct / n
    return avg_loss, accuracy


def train(group: str, freeze_mode: str, binary: bool):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    CKPT_DIR.mkdir(exist_ok=True)

    task = "binary" if binary else "severity"

    train_loader, val_loader = make_loaders(group, binary)
    model = build_model(device, freeze_mode=freeze_mode, binary=binary)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    scheduler = None  # no LR scheduling — simpler is better with small datasets

    trainable, total = count_params(model)
    print(f"Group: {group}  |  task: {task}  |  freeze: {freeze_mode}  |  trainable: {trainable:,} / {total:,} params")
    print(f"Train batches: {len(train_loader)}  |  Val incidents: {val_loader.dataset.num_incidents}")
    print()

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, loss_fn, device)
        if scheduler:
            scheduler.step()

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"train loss {train_loss:.4f}  acc {train_acc:.3f}  |  "
            f"val loss {val_loss:.4f}  acc {val_acc:.3f}",
            flush=True
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = CKPT_DIR / f"{group}_{task}_{freeze_mode}_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ saved checkpoint (val acc {val_acc:.3f})", flush=True)

    print(f"\nDone. Best val acc: {best_val_acc:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group",  choices=["inside", "outside"], required=True)
    parser.add_argument("--freeze", choices=["head", "layer4"], default="head",
                        help="head = fc only (~3K params); layer4 = layer4+fc (~24.9M params)")
    parser.add_argument("--binary", action="store_true",
                        help="train foul/no-foul (2 classes) instead of full severity (6 classes)")
    args = parser.parse_args()
    train(args.group, args.freeze, args.binary)
