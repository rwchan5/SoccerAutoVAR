import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_video

# Maps the severity string from our CSVs to an integer class index for the model
SEVERITY_TO_CLASS = {"": 0, "1.0": 1, "2.0": 2, "3.0": 3, "4.0": 4, "5.0": 5}
SEVERITY_TO_BINARY = {"": 0, "1.0": 1, "2.0": 1, "3.0": 1, "4.0": 1, "5.0": 1}
NUM_CLASSES = 6
NUM_CLASSES_BINARY = 2

# Normalization stats from Kinetics-400 — must match what R(2+1)D was pretrained with
KINETICS_MEAN = [0.43216, 0.39466, 0.37645]
KINETICS_STD  = [0.22803, 0.22145, 0.21690]

NUM_FRAMES = 16   # fixed input length R(2+1)D-18 expects
FRAME_SIZE = 112  # spatial resolution R(2+1)D-18 expects


class FoulDataset(Dataset):
    """
    Loads foul video clips from a split manifest CSV and returns tensors
    ready for R(2+1)D-18.

    Each item is a tuple: (frames, label, incident_idx)
      - frames:       FloatTensor of shape (3, 16, 112, 112)  [C, T, H, W]
      - label:        int class index  (0–5)
      - incident_idx: int row index in the CSV — used during evaluation to
                      group all clips of the same incident together for averaging

    Training mode  (train=True):  one item per incident, clip_0 only, random jitter
    Eval mode      (train=False): one item per (incident, clip) pair, fixed sampling
    """

    def __init__(self, csv_path, video_root, train: bool = True, binary: bool = False):
        self.video_root = Path(video_root)
        self.train = train
        self.label_map = SEVERITY_TO_BINARY if binary else SEVERITY_TO_CLASS

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        # Build flat list of (row, clip_path) and a parallel list of incident indices.
        # During eval, one incident can produce multiple items (one per clip).
        self.items: list[tuple[dict, Path]] = []
        self.incident_indices: list[int] = []

        for incident_idx, row in enumerate(rows):
            incident_dir = self.video_root / row["original_split"] / row["action_id"]
            if train:
                clip = incident_dir / "clip_0.mp4"
                if clip.exists():
                    self.items.append((row, clip))
                    self.incident_indices.append(incident_idx)
            else:
                clips = sorted(incident_dir.glob("clip_*.mp4"))
                for clip in clips:
                    self.items.append((row, clip))
                    self.incident_indices.append(incident_idx)

        self.num_incidents = len(rows)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        row, clip_path = self.items[idx]
        label = self.label_map[row["severity"]]
        incident_idx = self.incident_indices[idx]
        frames = self._load_frames(clip_path)
        return frames, label, incident_idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_frames(self, clip_path: Path) -> torch.Tensor:
        # read_video returns (T, H, W, C) uint8 — we only need the video tensor
        video, _, _ = read_video(str(clip_path), pts_unit="sec", output_format="THWC")

        indices = self._sample_indices(len(video))
        frames = video[indices]                          # (16, H, W, C)  uint8

        # Rearrange to (16, C, H, W), convert to float in [0, 1]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0

        # Resize spatial dims to 112×112 using bilinear interpolation.
        # interpolate expects (N, C, H, W) so we treat the 16 frames as a batch.
        frames = F.interpolate(
            frames, size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False
        )                                                # (16, C, 112, 112)

        # Apply Kinetics normalization channel-wise
        mean = torch.tensor(KINETICS_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(KINETICS_STD,  dtype=torch.float32).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        # R(2+1)D expects (C, T, H, W)
        return frames.permute(1, 0, 2, 3)

    def _sample_indices(self, total_frames: int) -> torch.Tensor:
        """
        Divide the clip into NUM_FRAMES equal segments, then pick one frame
        from each segment.

        Train: random position within each segment  → slight variation each epoch
        Eval:  center of each segment               → deterministic, reproducible
        """
        seg = total_frames / NUM_FRAMES
        if self.train:
            offsets = torch.rand(NUM_FRAMES) * seg
        else:
            offsets = torch.full((NUM_FRAMES,), seg / 2)

        starts  = torch.arange(NUM_FRAMES, dtype=torch.float32) * seg
        indices = (starts + offsets).long().clamp(0, total_frames - 1)
        return indices
