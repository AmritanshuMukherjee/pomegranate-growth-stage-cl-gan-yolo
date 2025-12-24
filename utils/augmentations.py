import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(dataset_cfg):
    image_size = dataset_cfg["image_size"]

    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,      # 🔥 KEY FIX
            clip=True,               # 🔥 CLAMP BOXES
        ),
    )


def get_val_transforms(dataset_cfg):
    image_size = dataset_cfg["image_size"]

    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            clip=True,
        ),
    )
