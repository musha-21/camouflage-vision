
import torch
import traceback

import config
import data_model

print("Starting one-batch training test...", flush=True)

try:
    model = config.model
    model.to(config.DEVICE)
    model.train()

    # Get one complete batch from the training DataLoader.
    images, masks = next(iter(data_model.train_loader))

    images = images.to(config.DEVICE)
    masks = masks.to(config.DEVICE)

    if masks.ndim == 4:
        if masks.shape[1] == len(config.CLASSES):
            masks = torch.argmax(masks, dim=1)
        elif masks.shape[1] == 1:
            masks = masks.squeeze(1)

    masks = masks.long()

    print("Configured batch size:", config.BATCH_SIZE, flush=True)
    print("Actual image shape:", images.shape, flush=True)
    print("Actual mask shape:", masks.shape, flush=True)

    config.optimizer.zero_grad()

    print("Running forward pass...", flush=True)
    outputs = model(images)

    print("Calculating loss...", flush=True)
    loss = config.loss(outputs, masks)
    print("Loss:", loss.item(), flush=True)

    print("Running backward pass...", flush=True)
    loss.backward()

    print("Updating model weights...", flush=True)
    config.optimizer.step()

    print("SUCCESS: One complete batch trained!", flush=True)

except Exception:
    traceback.print_exc()
