import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
import config
import utils
from utils import get_non_repeating_numbers


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def save_visualization(dataset, folder_name, enabled=False):
    if (enabled == False):
        return
    utils.create_subfolder_in_date_folder("vizualization_of_data_set")
    utils.create_subfolder_in_date_folder("vizualization_of_data_set\\" + folder_name)
    images_count = len(dataset)
    for n in range(images_count):
        image, mask = dataset[n]

        normal_image = Image.fromarray(np.uint8(image))
        animals = Image.fromarray(np.uint8(mask[..., 0].squeeze()))
        masking_background = Image.fromarray(np.uint8(mask[..., 1].squeeze()))
        nonmasking_background = Image.fromarray(np.uint8(mask[..., 2].squeeze()))
        foreground_attention = Image.fromarray(np.uint8(mask[..., 3].squeeze()))

        normal_image = normal_image.resize((normal_image.width * 2, normal_image.height * 2))
        new_image = Image.new('RGB', (normal_image.width * 2, normal_image.height))

        # Add red border of 1px to each image
        normal_image = ImageOps.expand(normal_image, border=1, fill='red')
        animals = ImageOps.expand(animals, border=1, fill='red')
        masking_background = ImageOps.expand(masking_background, border=1, fill='red')
        nonmasking_background = ImageOps.expand(nonmasking_background, border=1, fill='red')
        foreground_attention = ImageOps.expand(foreground_attention, border=1, fill='red')

        new_image.paste(normal_image, (0, 0))

        new_image.paste(animals, (normal_image.width, 0))
        new_image.paste(foreground_attention, (int(normal_image.width + normal_image.width / 2), 0))

        new_image.paste(masking_background, (normal_image.width, int(normal_image.height / 2)))
        new_image.paste(nonmasking_background,
                        (int(normal_image.width + normal_image.width / 2), int(normal_image.height / 2)))

        path = os.path.join(config.CURRENT_PATH, os.path.join("vizualization_of_data_set", os.path.join(folder_name,
                                                                                                        'merged_image' + str(
                                                                                                            n) + '.png')))
        new_image.save(path)


train_scores = []
valid_scores = []
test_scores = []

train_losses = []
valid_losses = []
test_losses = []

first_launch = True
fig_number = 0


def update_plot(train_logs, valid_logs, test_log, enabled=True):
    if not enabled:
        return False

    if plt.get_fignums():
        fig = plt.figure(1, figsize=(16, 8), dpi=80)
        fig.clf()
        axs = fig.subplots(1, 2)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=80)

    train_score = config.calculate_score(train_logs)
    valid_score = config.calculate_score(valid_logs)
    test_score = 0  # config.calculate_score(test_log)

    train_loss = config.calculate_loss(train_logs)
    valid_loss = config.calculate_loss(valid_logs)
    test_loss = 0  # config.calculate_loss(test_log)

    train_scores.append(train_score)
    valid_scores.append(valid_score)
    test_scores.append(test_score)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    test_losses.append(test_loss)

    axs[0].plot(train_scores, color='blue', label='Train score')
    axs[0].plot(valid_scores, color='orange', label='Valid score')
    axs[0].plot(test_scores, color='red', label='Train score')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Score')
    axs[0].legend(loc='upper left')
    axs[0].set_title("Train score: {}\nValid score: {}\nTest score: {}".format(round(train_score, 6),
                                                                               round(valid_score, 6),
                                                                               round(test_score, 6)))

    axs[1].plot(train_losses, color='blue', label='Train loss')
    axs[1].plot(valid_losses, color='orange', label='Valid loss')
    axs[1].plot(test_losses, color='red', label='Train loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper right')
    axs[1].set_title("Train loss: {}\nValid loss: {}\nTest loss: {}".format(round(train_loss, 3),
                                                                            round(valid_loss, 3),
                                                                            round(test_loss, 3)))

    fig.canvas.flush_events()
    plt.pause(5)
    plt.tight_layout()
    plt.savefig(os.path.join(config.CURRENT_PATH, "fig.png"))


def save_results(test_visualize, test_dataset, count=10):
    """Save original images, ground-truth masks, and U-Net predictions."""

    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    output_dir = os.path.join(config.CURRENT_PATH, "test_result")
    os.makedirs(output_dir, exist_ok=True)

    model = config.model
    model.eval()

    colors = np.asarray(config.COLORS, dtype=np.uint8)
    device = config.DEVICE

    indices = get_non_repeating_numbers(
        len(test_dataset), min(count, len(test_dataset))
    )

    for i in indices:
        image, gt_mask = test_dataset[i]

        # Convert input image to a tensor.
        if torch.is_tensor(image):
            x = image.detach().clone().float()
        else:
            x = torch.as_tensor(np.asarray(image)).float()

        # Convert HWC image to CHW if necessary.
        if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
            if x.shape[0] not in (1, 3, 4):
                x = x.permute(2, 0, 1)

        # Prepare ground-truth class IDs.
        if torch.is_tensor(gt_mask):
            gt = gt_mask.detach().cpu().numpy()
        else:
            gt = np.asarray(gt_mask)

        gt = np.squeeze(gt)

        # Support either class-ID masks or one-hot masks.
        if gt.ndim == 3 and gt.shape[-1] == len(colors):
            gt = np.argmax(gt, axis=-1)
        elif gt.ndim == 3 and gt.shape[0] == len(colors):
            gt = np.argmax(gt, axis=0)

        if gt.ndim != 2:
            raise ValueError(
                f"Unexpected ground-truth mask shape: {gt.shape}"
            )

        # Predict class ID for every pixel.
        with torch.inference_mode():
            logits = model(x.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1)[0].cpu().numpy()

        # Convert class IDs to RGB colors.
        def colorize(mask):
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            valid = (mask >= 0) & (mask < len(colors))
            rgb[valid] = colors[mask[valid].astype(int)]
            return rgb

        gt_rgb = colorize(gt)
        pred_rgb = colorize(pred)

        # Animal class is class 0 in this project.
        gt_animal = (gt == 0).astype(np.uint8) * 255
        pred_animal = (pred == 0).astype(np.uint8) * 255

        # Convert input image to a displayable RGB image.
        image_vis = x.detach().cpu().numpy().transpose(1, 2, 0)

        # Undo ImageNet normalization when the image is normalized.
        if image_vis.min() < 0 or image_vis.max() > 1:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_vis = image_vis * std + mean

        image_vis = (np.clip(image_vis, 0, 1) * 255).astype(np.uint8)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        panels = [
            (image_vis, "Original Image", None),
            (gt_rgb, "Ground Truth: All Classes", None),
            (pred_rgb, "Prediction: All Classes", None),
            (gt_animal, "Ground Truth: Animal", "gray"),
            (pred_animal, "Prediction: Animal", "gray"),
            (
                np.abs(
                    gt_animal.astype(np.int16)
                    - pred_animal.astype(np.int16)
                ).astype(np.uint8),
                "Animal Mask Difference",
                "gray",
            ),
        ]

        for ax, (picture, title, cmap) in zip(axes.flat, panels):
            ax.imshow(picture, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")

        fig.tight_layout()

        if hasattr(test_dataset, "get_name"):
            filename = test_dataset.get_name(i)
        else:
            filename = f"test_{i}"

        filename = os.path.splitext(os.path.basename(str(filename)))[0]
        save_path = os.path.join(output_dir, filename + "_result.png")

        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved prediction: {save_path}")

    print(f"All visualizations saved in: {os.path.abspath(output_dir)}")
