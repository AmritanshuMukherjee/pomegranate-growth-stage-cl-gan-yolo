import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(dataset_cfg):
    img_size = dataset_cfg["image_size"]

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_val_transforms(dataset_cfg):
    img_size = dataset_cfg["image_size"]

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
