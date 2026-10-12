
import random
import numpy as np
import ssl
import os
import datetime
import gc

from multiprocessing import freeze_support

import torch
import config


def setup_env():
    """Initialize the training environment."""
    freeze_support()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ssl._create_default_https_context = ssl._create_unverified_context

    config.save_config(config.CURRENT_PATH)

    if torch.cuda.is_available() and str(config.DEVICE).startswith("cuda"):
        try:
            torch.cuda.empty_cache()
            print(
                torch.cuda.memory_summary(
                    device=config.DEVICE,
                    abbreviated=True,
                )
            )
        except (KeyError, RuntimeError, AssertionError) as error:
            print(f"CUDA memory summary unavailable: {error}")
    else:
        print(
            "CUDA is unavailable or not selected. "
            "Skipping GPU memory summary."
        )

    gc.collect()


def create_current_folder():
    """Create a timestamped folder for this training run."""
    root_dir = os.getcwd()
    results_dir = os.path.join(root_dir, "!Results")
    os.makedirs(results_dir, exist_ok=True)

    current_time = datetime.datetime.now()
    subfolder_name = current_time.strftime("%d-%m_%H-%M-%S")
    subfolder_dir = os.path.join(results_dir, subfolder_name)

    os.makedirs(subfolder_dir, exist_ok=True)

    return subfolder_dir


def denormalize(x):
    """Scale image values to the range 0..1."""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)

    if x_max == x_min:
        return np.zeros_like(x, dtype=np.float32)

    x = (x - x_min) / (x_max - x_min)
    return x.clip(0, 1)


def crop_to_divisible_by_32(image):
    """Crop image dimensions to multiples of 32."""
    h, w = image.shape[:2]
    h_new = h - h % 32
    w_new = w - w % 32

    return image[:h_new, :w_new]


def to_tensor(x, **kwargs):
    import numpy as np

    x = np.asarray(x)

    if x.ndim == 3:
        # Convert image from HWC to CHW.
        return x.transpose(2, 0, 1).astype("float32")

    if x.ndim == 2:
        # Keep segmentation mask as H x W class IDs.
        return x.astype("int64")

    raise ValueError(
        f"Unexpected input shape for to_tensor: {x.shape}"
    )


def save_model():
    """Save the model's weights to the current results folder."""
    if not getattr(config, "IS_MODEL_SAVED", True):
        print("Model saving is disabled.", flush=True)
        return

    if not config.CURRENT_PATH:
        raise RuntimeError(
            "CURRENT_PATH is not set. Create a training folder first."
        )

    os.makedirs(config.CURRENT_PATH, exist_ok=True)

    model_path = os.path.join(config.CURRENT_PATH, "model.pth")
    temp_path = model_path + ".tmp"

    try:
        torch.save(config.model.state_dict(), temp_path)
        os.replace(temp_path, model_path)
        print(f"Model weights saved: {model_path}", flush=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_model(model_path=None):
    """Load saved weights into the configured model."""
    if model_path is None:
        model_path = os.path.join(config.CURRENT_PATH, "model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    state_dict = torch.load(
        model_path,
        map_location=config.DEVICE,
        weights_only=True,
    )

    config.model.load_state_dict(state_dict)
    config.model.to(config.DEVICE)

    print(f"Model weights loaded: {model_path}", flush=True)

    return config.model


def continue_training(date="", enabled=False):
    """Load saved weights when resuming an existing training run."""
    if not enabled:
        return False

    results_dir = os.path.join(os.getcwd(), "!Results")
    subfolder_dir = os.path.join(results_dir, date)
    model_path = os.path.join(subfolder_dir, "model.pth")

    if not os.path.isdir(subfolder_dir):
        print("Training folder does not exist:", subfolder_dir)
        return False

    if not os.path.isfile(model_path):
        print("Could not locate model:", model_path)
        return False

    load_model(model_path)
    config.CURRENT_PATH = subfolder_dir

    print("##############################")
    print("#      RETRAINING MODEL      #")
    print("##############################")

    return True


def create_subfolder_in_date_folder(name):
    """Create a subfolder inside the current training folder."""
    folder = os.path.join(config.CURRENT_PATH, name)
    os.makedirs(folder, exist_ok=True)
    return folder


def config_line(param, value):
    """Format a configuration entry."""
    return f"{param} = {value}\n"


def get_non_repeating_numbers(test_count, amount_of_numbers):
    """Return unique random indices."""
    if test_count <= 0:
        return []

    amount_of_numbers = max(
        0,
        min(amount_of_numbers, test_count),
    )

    return random.sample(range(test_count), amount_of_numbers)
