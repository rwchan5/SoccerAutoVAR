"""
Cross-evaluation script.

Loads each trained checkpoint and evaluates it on both its own test set
and the opposite group's test set. Reports accuracy and majority baseline
for each combination so the cross-domain gap is clear.

Usage (from model/ directory):
    conda run -n SoccerNet python evaluate.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import FoulDataset
from model import build_model

CKPT_DIR   = Path(__file__).parent / "checkpoints"
DATA_DIR   = Path(__file__).parent / "data"
VIDEO_ROOT = Path(__file__).parent.parent / "SoccerNetData" / "mvfouls"


def run_eval(model, csv_path, binary, device):
    """Evaluate model on a test CSV. Returns (correct, total, no_foul_count, foul_count)."""
    ds     = FoulDataset(csv_path, VIDEO_ROOT, train=False, binary=binary)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    incident_scores = defaultdict(list)
    incident_labels = {}

    model.eval()
    with torch.no_grad():
        for frames, labels, idxs in loader:
            frames = frames.to(device)
            probs  = torch.softmax(model(frames), dim=1)
            for idx, prob, label in zip(idxs.tolist(), probs, labels.tolist()):
                incident_scores[idx].append(prob.cpu())
                incident_labels[idx] = label

    correct  = 0
    no_foul  = sum(1 for l in incident_labels.values() if l == 0)
    foul     = sum(1 for l in incident_labels.values() if l == 1)

    for idx, scores in incident_scores.items():
        avg  = torch.stack(scores).mean(dim=0)
        pred = avg.argmax().item()
        correct += int(pred == incident_labels[idx])

    return correct, len(incident_scores), no_foul, foul


def print_result(label, correct, total, no_foul, foul):
    baseline = max(no_foul, foul)
    acc      = correct / total
    base_acc = baseline / total
    gap      = acc - base_acc
    print(f"  {label}")
    print(f"    Distribution : {no_foul} no-foul / {foul} foul")
    print(f"    Baseline     : {baseline}/{total} = {base_acc:.3f}")
    print(f"    Model acc    : {correct}/{total} = {acc:.3f}  ({gap:+.3f} vs baseline)")
    print()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    configs = [
        ("inside",  "inside_binary_head_best.pt"),
        ("outside", "outside_binary_head_best.pt"),
    ]

    for trained_on, ckpt_name in configs:
        opposite = "outside" if trained_on == "inside" else "inside"
        print(f"{'='*60}")
        print(f"Model trained on: {trained_on.upper()}")
        print(f"{'='*60}")

        model = build_model(device, freeze_mode="head", binary=True)
        model.load_state_dict(torch.load(CKPT_DIR / ckpt_name, map_location=device))

        # Own test set
        correct, total, nf, f = run_eval(
            model, DATA_DIR / f"{trained_on}_test.csv", binary=True, device=device
        )
        print_result(f"Own test set ({trained_on})", correct, total, nf, f)

        # Opposite test set
        correct, total, nf, f = run_eval(
            model, DATA_DIR / f"{opposite}_test.csv", binary=True, device=device
        )
        print_result(f"Cross test set ({opposite})", correct, total, nf, f)


if __name__ == "__main__":
    main()
