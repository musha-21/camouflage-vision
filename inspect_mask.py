
import os
import cv2
import numpy as np

# Training mask directory
MASK_FOLDER = r".\!Dataset\Dataset1\Train\Masks"

# Number of masks to inspect
NUMBER_OF_MASKS = 10

# Allowed image extensions
EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def inspect_mask(mask_path):
    """Print the most common RGB colors in a mask."""

    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

    if mask is None:
        print(f"ERROR: Could not read {mask_path}")
        return

    # OpenCV reads BGR; convert to RGB.
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    colors, counts = np.unique(
        mask.reshape(-1, 3),
        axis=0,
        return_counts=True,
    )

    order = np.argsort(counts)[::-1]

    print(f"\nFile: {os.path.basename(mask_path)}")
    print("Dimensions:", mask.shape)
    print("Unique colors:", len(colors))
    print("Most common RGB colors:")

    for index in order[:10]:
        color = colors[index].tolist()
        count = int(counts[index])

        print(f"  RGB {color}: {count} pixels")


def main():
    if not os.path.isdir(MASK_FOLDER):
        raise FileNotFoundError(
            f"Mask folder not found: {MASK_FOLDER}"
        )

    files = sorted(
        filename
        for filename in os.listdir(MASK_FOLDER)
        if filename.lower().endswith(EXTENSIONS)
    )

    if not files:
        print("No mask images found.")
        return

    print("Mask folder:", MASK_FOLDER)
    print("Total masks:", len(files))
    print(f"Inspecting up to {NUMBER_OF_MASKS} masks...")

    for filename in files[:NUMBER_OF_MASKS]:
        mask_path = os.path.join(MASK_FOLDER, filename)
        inspect_mask(mask_path)

    print("\nInspection complete.")
    print("Configured class colors:")
    print("  Blue:   RGB [0, 0, 255]")
    print("  Green:  RGB [0, 255, 0]")
    print("  Red:    RGB [255, 0, 0]")
    print("  White:  RGB [255, 255, 255]")
    print("\nBlack pixels need to be interpreted before training.")


if __name__ == "__main__":
    main()
