
import torch
import config

import data_model.datasets as datasets


def run_epoch(dataloader, training=False):
    """Run one training or validation epoch."""

    mode = "TRAINING" if training else "VALIDATION"
    print(f"\n[DEBUG] Entered {mode} epoch", flush=True)

    model = config.model
    loss_fn = config.loss
    device = config.DEVICE

    print("[DEBUG] Model:", type(model).__name__, flush=True)
    print("[DEBUG] Device:", device, flush=True)
    print("[DEBUG] Batches:", len(dataloader), flush=True)

    model.to(device)
    model.train() if training else model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        print(
            f"\n[DEBUG] Batch {batch_idx + 1}/{len(dataloader)}",
            flush=True,
        )

        images = images.to(device)
        masks = masks.to(device)

        print("[DEBUG] Image shape:", tuple(images.shape), flush=True)
        print("[DEBUG] Mask shape:", tuple(masks.shape), flush=True)

        # Convert one-hot masks to class IDs.
        print("[DEBUG] Starting mask conversion", flush=True)

        if masks.ndim == 4:
            if masks.shape[1] == len(config.CLASSES):
                masks = torch.argmax(masks, dim=1)
            elif masks.shape[1] == 1:
                masks = masks.squeeze(1)

        masks = masks.long()

        print("[DEBUG] Mask conversion complete", flush=True)
        print("[DEBUG] Mask dtype:", masks.dtype, flush=True)
        print("[DEBUG] Mask values:", torch.unique(masks), flush=True)

        # Clear gradients before training.
        if training:
            config.optimizer.zero_grad()

        print("[DEBUG] Ready for forward pass", flush=True)

        # Forward pass and loss calculation.
        with torch.set_grad_enabled(training):
            print("[DEBUG] Starting model forward pass...", flush=True)
            outputs = model(images)
            print("[DEBUG] Forward pass completed!", flush=True)

            print(
                "[DEBUG] Output shape:",
                tuple(outputs.shape),
                flush=True,
            )

            loss = loss_fn(outputs, masks)

            # Backpropagation and parameter update.
            if training:
                loss.backward()
                config.optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Calculate IoU.
        with torch.no_grad():
            iou = config.metrics[0](outputs.detach(), masks)

            iou_value = (
                iou.item()
                if torch.is_tensor(iou)
                else float(iou)
            )

            total_iou += iou_value * batch_size

        print(
            f"[DEBUG] Loss: {loss.item():.4f}, "
            f"IoU: {iou_value:.4f}",
            flush=True,
        )

    if total_samples == 0:
        raise RuntimeError(
            f"{mode} epoch processed zero samples."
        )

    logs = {
        "cross_entropy_loss": total_loss / total_samples,
        "iou_score": total_iou / total_samples,
    }

    print(f"[DEBUG] {mode} completed: {logs}", flush=True)

    return logs


def train_one_epoch(dataloader):
    """Train the model for one epoch."""
    print("[DEBUG] Calling train_one_epoch()", flush=True)
    return run_epoch(dataloader, training=True)


def validate_one_epoch(dataloader):
    """Validate the model for one epoch."""
    print("[DEBUG] Calling validate_one_epoch()", flush=True)
    return run_epoch(dataloader, training=False)


def evaluate_test_data():
    """Evaluate the model on the test dataset."""
    print("[DEBUG] Evaluating test dataset...", flush=True)

    test_logs = validate_one_epoch(datasets.test_dataloader)

    print("Test loss:", test_logs["cross_entropy_loss"])
    print("Test IoU:", test_logs["iou_score"])

    return test_logs
