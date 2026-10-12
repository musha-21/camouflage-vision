
import os
import config
import utils
import data_model.datasets as datasets
from utils import visualizers

# Change this path if your trained model is in another folder.
checkpoint_path = r".\!Results\11-10_18-08-56\model.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. "
        "Please check the model.pth path."
    )

# Load the trained model.
utils.load_model(checkpoint_path)
config.model.eval()

# Get the existing test dataset.
test_dataset = datasets.test_dataloader.dataset

# Save visual comparisons for 10 test images.
visualizers.save_results(
    test_visualize=None,
    test_dataset=test_dataset,
    count=10
)

print("Visualization completed!")

from PIL import Image
import os

image_path = os.path.abspath(
    r"test_result\b9qvqsag_result.png"
)

Image.open(image_path).show()