import argparse
import csv
import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
LABELS_CSV = ROOT / "labeler" / "box_labels.csv"
DATA_DIR = ROOT / "SoccerNetData" / "mvfouls"
OUT_DIR = Path(__file__).parent / "data"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def load_annotations():
    anns = {}
    for split in ("train", "valid", "test"):
        path = DATA_DIR / split / "annotations.json"
        with open(path) as f:
            data = json.load(f)["Actions"]
        for action_num, entry in data.items():
            anns[(split, action_num)] = entry.get("Severity", "")
    return anns


def load_box_labels(anns):
    rows = []
    with open(LABELS_CSV) as f:
        for row in csv.DictReader(f):
            action_num = str(int(row["action_id"].replace("action_", "")))
            severity = anns.get((row["split"], action_num), "")
            rows.append(
                {
                    "original_split": row["split"],
                    "action_id": row["action_id"],
                    "box_location": row["box_location"],
                    "severity": severity,
                }
            )
    return rows


def stratify_key(severity):
    if severity in ("4.0", "5.0"):
        return "4_5"  # group red-card-level together
    if severity == "2.0":
        return "2_3"  # group with 3.0 since 2.0 has very few samples
    return severity


def binary_key(severity):
    return "foul" if severity != "" else "no_foul"


def split_dataset(rows, seed):
    keys = [stratify_key(r["severity"]) for r in rows]
    train_rows, temp_rows = train_test_split(
        rows, test_size=1 - TRAIN_RATIO, random_state=seed, stratify=keys
    )
    # Stratify val/test on binary label — simpler grouping avoids 1-member failures
    binary_keys = [binary_key(r["severity"]) for r in temp_rows]
    try:
        val_rows, test_rows = train_test_split(
            temp_rows, test_size=0.5, random_state=seed, stratify=binary_keys
        )
    except ValueError:
        # Fallback if one binary class has only 1 member in temp
        val_rows, test_rows = train_test_split(
            temp_rows, test_size=0.5, random_state=seed
        )
    return train_rows, val_rows, test_rows

def write_csv(rows, path):
    fields = ["original_split", "action_id", "box_location", "severity"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    anns = load_annotations()
    all_rows = load_box_labels(anns)

    inside = [r for r in all_rows if r["box_location"] == "inside"]
    outside = [r for r in all_rows if r["box_location"] == "outside"]
    outside_foul    = [r for r in outside if r["severity"] != ""]
    outside_no_foul = [r for r in outside if r["severity"] == ""]
    n = len(inside)
    n_no_foul = round(n * len(outside_no_foul) / len(outside))  # proportional
    n_foul    = n - n_no_foul
    outside_sample = (
        random.sample(outside_no_foul, min(n_no_foul, len(outside_no_foul))) +
        random.sample(outside_foul,    min(n_foul,    len(outside_foul)))
    )
    for group_name, group_rows in [("inside", inside), ("outside", outside_sample)]:
        print(f"\n--- {group_name} ---")
        train, val, test = split_dataset(group_rows, seed=args.seed)
        print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
        write_csv(train, OUT_DIR / f"{group_name}_train.csv")
        write_csv(val,   OUT_DIR / f"{group_name}_val.csv")
        write_csv(test,  OUT_DIR / f"{group_name}_test.csv")

if __name__ == "__main__":
    main()
