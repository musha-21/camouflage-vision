
import config
import utils

from data_model.dataset_class import Dataset
from torch.utils.data import DataLoader


# Common settings
BATCH_SIZE = config.BATCH_SIZE
NUM_WORKERS = 0


# Training dataset: randomized augmentation
train_dataset = Dataset(
    config.x_train_dir,
    config.y_train_dir,
    augmentation=utils.get_training_augmentation(),
    preprocessing=utils.get_preprocessing(config.preprocessing_fn),
    classes=config.CLASSES,
)


# Validation dataset: fixed-size input
valid_dataset = Dataset(
    config.x_valid_dir,
    config.y_valid_dir,
    augmentation=utils.get_validation_augmentation(),
    preprocessing=utils.get_preprocessing(config.preprocessing_fn),
    classes=config.CLASSES,
)


# Test dataset: fixed-size input
test_dataset = Dataset(
    config.x_test_dir,
    config.y_test_dir,
    augmentation=utils.get_test_augmentation(),
    preprocessing=utils.get_preprocessing(config.preprocessing_fn),
    classes=config.CLASSES,
)


# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)


# Visualization datasets (no augmentation or preprocessing)
train_visualize = Dataset(
    config.x_train_dir,
    config.y_train_dir,
    classes=config.CLASSES,
)

valid_visualize = Dataset(
    config.x_valid_dir,
    config.y_valid_dir,
    classes=config.CLASSES,
)

test_visualize = Dataset(
    config.x_test_dir,
    config.y_test_dir,
    classes=config.CLASSES,
)
