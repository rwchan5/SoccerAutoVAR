import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

from dataset import NUM_CLASSES, NUM_CLASSES_BINARY


def build_model(device: torch.device, freeze_mode: str = "layer4", binary: bool = False) -> nn.Module:
    # load R(2+1)D-18 pretrained on Kinetics-400, and replace the classification head

    #freeze_mode options:  "head"  freeze everything except fc or "layer4" freeze everything except layer4 + fc 

    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    num_classes = NUM_CLASSES_BINARY if binary else NUM_CLASSES
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.fc.parameters():
        param.requires_grad_(True)
    if freeze_mode == "layer4":
        for param in model.layer4.parameters():
            param.requires_grad_(True)
    return model.to(device)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
