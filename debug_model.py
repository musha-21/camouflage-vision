
import torch
import traceback

import config
import data_model

print("Loading model...", flush=True)

try:
    model = config.model
    model.to("cpu")
    model.eval()

    images, masks = next(iter(data_model.train_loader))
    images = images[:1].to("cpu")

    print("Input shape:", tuple(images.shape), flush=True)
    print("Running model...", flush=True)

    with torch.no_grad():
        outputs = model(images)

    print("Forward pass successful!", flush=True)
    print("Output shape:", tuple(outputs.shape), flush=True)

except Exception:
    traceback.print_exc()

print("Debug test finished.", flush=True)
