
import albumentations as A


IMAGE_SIZE = 544


def get_training_augmentation():
    return A.Compose([
        A.LongestMaxSize(
            max_size=IMAGE_SIZE,
            interpolation=1,
            mask_interpolation=0,
        ),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=0,
            fill=0,
            fill_mask=255,
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.1,
            rotate_limit=10,
            shift_limit=0.1,
            p=0.7,
            border_mode=0,
            fill=0,
            fill_mask=255,
        ),
        A.RandomCrop(
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        ),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
            ),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
            ),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.3),
        A.GaussNoise(p=0.2),
    ])


def get_validation_augmentation():
    return A.Compose([
        A.LongestMaxSize(
            max_size=IMAGE_SIZE,
            interpolation=1,
            mask_interpolation=0,
        ),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=0,
            fill=0,
            fill_mask=255,
        ),
    ])


def get_test_augmentation():
    return get_validation_augmentation()
