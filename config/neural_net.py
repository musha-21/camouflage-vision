
import os

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet

from config import COLORS
from utils import config_line


# Segmentation classes
CLASSES = [
    "animal",
    "maskingbackground",
    "nonmaskingbackground",
    "nonmaskingforegroundattention",
]

# Class weights
WEIGHTS = torch.tensor([1.0, 0.05, 0.03, 0.01])

# Model configuration
ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = None  # Raw logits for CrossEntropyLoss
lr = 0.0001
BATCH_SIZE = 2


# Create the U-Net segmentation model
model = Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    in_channels=3,
    activation=ACTIVATION,
)

def predict_mask(image_tensor):
    model.eval()

    with torch.inference_mode():
        logits = model(image_tensor)
        predictions = torch.argmax(logits, dim=1)

    return predictions

# Weighted multiclass cross-entropy loss
loss = torch.nn.CrossEntropyLoss(
    weight=WEIGHTS,
    ignore_index=255,
)


# Multiclass Intersection over Union (IoU)

def iou_score(preds, targets, eps=1e-7):
    """Mean multiclass IoU, excluding ignored pixels."""

    pred_classes = torch.argmax(preds, dim=1)

    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    elif targets.ndim == 4 and targets.shape[1] == len(CLASSES):
        targets = torch.argmax(targets, dim=1)

    targets = targets.long()

    # Ignore void pixels labeled 255.
    valid = targets != 255
    pred_classes = pred_classes[valid]
    targets = targets[valid]

    if targets.numel() == 0:
        return preds.new_tensor(0.0)

    ious = []

    for cls in range(len(CLASSES)):
        pred_mask = pred_classes == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union.item() > 0:
            ious.append(intersection / (union + eps))

    if not ious:
        return preds.new_tensor(0.0)

    return torch.stack(ious).mean()



metrics = [iou_score]


# Adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
)


# Encoder preprocessing
preprocessing_fn = smp.encoders.get_preprocessing_fn(
    ENCODER,
    ENCODER_WEIGHTS,
)


def calculate_score(logs):
    return logs["iou_score"]


def calculate_loss(logs):
    return logs["cross_entropy_loss"]


def save_config(current_path=""):
    """Save model and training configuration to config.cfg."""

    os.makedirs(current_path or ".", exist_ok=True)
    config_path = os.path.join(current_path, "config.cfg")

    with open(config_path, "w", encoding="utf-8") as f:
        for i, class_name in enumerate(CLASSES):
            f.write(f"#Class_{i}\n")
            f.write(config_line("Name", class_name))
            f.write(config_line("Color", str(COLORS[i])))
            f.write(config_line("LossWeight", str(WEIGHTS[i].item())))
            f.write("\n")

        for i, metric in enumerate(metrics):
            f.write(f"#Metric_{i}\n")
            f.write(config_line("Name", metric.__name__))
            f.write("\n")

        f.write(config_line("Encoder", ENCODER))
        f.write(config_line("EncoderWeights", ENCODER_WEIGHTS))
        f.write(config_line("Activation", str(ACTIVATION)))
        f.write(config_line("Model", model.__class__.__name__))
        f.write(config_line("LossFunction", loss.__class__.__name__))
        f.write(config_line("OptimizerFunction", optimizer.__class__.__name__))
        f.write(config_line("LearningRate", lr))
        f.write(config_line("BatchSize", BATCH_SIZE))

