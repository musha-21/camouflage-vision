
import os
import cv2
import numpy as np

import config
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """Dataset for four-class camouflaged object segmentation."""

    IGNORE_INDEX = 255

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"Image directory not found: {images_dir}"
            )

        if not os.path.isdir(masks_dir):
            raise FileNotFoundError(
                f"Mask directory not found: {masks_dir}"
            )

        extensions = (
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
        )

        image_map = {
            os.path.splitext(name)[0]: name
            for name in os.listdir(images_dir)
            if name.lower().endswith(extensions)
        }

        mask_map = {
            os.path.splitext(name)[0]: name
            for name in os.listdir(masks_dir)
            if name.lower().endswith(extensions)
        }

        common_names = sorted(set(image_map) & set(mask_map))

        if not common_names:
            raise RuntimeError(
                f"No matching image-mask pairs found:\n"
                f"{images_dir}\n{masks_dir}"
            )

        if len(image_map) != len(common_names):
            print(
                f"Warning: {len(image_map) - len(common_names)} "
                "images have no matching masks."
            )

        if len(mask_map) != len(common_names):
            print(
                f"Warning: {len(mask_map) - len(common_names)} "
                "masks have no matching images."
            )

        self.images_fps = [
            os.path.join(images_dir, image_map[name])
            for name in common_names
        ]
        self.masks_fps = [
            os.path.join(masks_dir, mask_map[name])
            for name in common_names
        ]
        self.ids = common_names

        if classes is None:
            classes = config.CLASSES

        self.class_values = [
            config.CLASSES.index(cls.lower())
            for cls in classes
        ]

        self.class_colors = [
            np.asarray(config.COLORS[index], dtype=np.uint8)
            for index in self.class_values
        ]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        print(
            f"Loaded {len(self.images_fps)} image-mask pairs "
            f"from {images_dir}"
        )

    def __len__(self):
        return len(self.images_fps)

    def get_name(self, i):
        return self.ids[i]

    def __getitem__(self, i):
        image_path = self.images_fps[i]
        mask_path = self.masks_fps[i]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image and mask dimensions differ: {image_path}"
            )

        # Crop to dimensions divisible by 32.
        height, width = image.shape[:2]
        height = (height // 32) * 32
        width = (width // 32) * 32

        if height == 0 or width == 0:
            raise ValueError(f"Image is too small: {image_path}")

        image = image[:height, :width]
        mask = mask[:height, :width]

        # Initialize every pixel as ignored.
        class_mask = np.full(
            (height, width),
            self.IGNORE_INDEX,
            dtype=np.uint8,
        )

        # Convert configured colors into class IDs.
        # Convert configured colors into class IDs in small chunks.
        chunk_size = 128

        for class_id, color in enumerate(self.class_colors):
            for start in range(0, height, chunk_size):
                end = min(start + chunk_size, height)
                chunk = mask[start:end]

                matches = (
                    (chunk[:, :, 0] == color[0])
                    & (chunk[:, :, 1] == color[1])
                    & (chunk[:, :, 2] == color[2])
                )

                class_mask[start:end][matches] = class_id

        # Black pixels and other unrecognized colors remain ignored.
        if self.augmentation is not None:
            sample = self.augmentation(
                image=image,
                mask=class_mask,
            )
            image = sample["image"]
            class_mask = sample["mask"]

        if self.preprocessing is not None:
            sample = self.preprocessing(
                image=image,
                mask=class_mask,
            )
            image = sample["image"]
            class_mask = sample["mask"]

        # Convert image to a CHW float tensor if preprocessing
        # has not already done so.
        if isinstance(image, np.ndarray):
            image = np.ascontiguousarray(image)

            if image.ndim == 3 and image.shape[-1] == 3:
                image = image.transpose(2, 0, 1)

            import torch
            image = torch.from_numpy(image).float()

        import torch
        class_mask = torch.as_tensor(
            np.ascontiguousarray(class_mask),
            dtype=torch.long,
        )

        return image, class_mask
