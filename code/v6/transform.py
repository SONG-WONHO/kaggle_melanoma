from albumentations.pytorch import ToTensor
from albumentations import *
import torchtoolbox.transform as transforms
import cv2


### torchtoolbox
def transform_v0(config):
    """ https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet/data?scriptVersionId=35726268

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
        transforms.Cutout(scale=(0.05, 0.007), value=(0, 0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, test_transforms


### albumentations
def transform_v1(config):
    """ default transforms

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    test_transforms = Compose([
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    return train_transforms, test_transforms


def transform_v2(config):
    """ Flip, Rotate

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        Flip(p=1),
        RandomRotate90(p=1),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    test_transforms = Compose([
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    return train_transforms, test_transforms


def transform_v3(config):
    """ Flip, ShiftScaleRotate

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        Flip(p=1),
        ShiftScaleRotate(p=1),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    test_transforms = Compose([
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    return train_transforms, test_transforms


def transform_v4(config):
    """ Flip, Rotate, RandomBrightnessContrast

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        Flip(p=1),
        RandomRotate90(p=1),
        RandomBrightnessContrast(0.1, 0.1, p=1),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    test_transforms = Compose([
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    return train_transforms, test_transforms


def transform_v5(config):
    """ Flip, Rotate, RandomBrightnessContrast

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        Flip(p=1),
        RandomRotate90(p=1),
        RandomBrightnessContrast(0.1, 0.1, p=1),
        HueSaturationValue(10, 10, 10, p=1),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    test_transforms = Compose([
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    return train_transforms, test_transforms


def get_transform(config):
    try:
        name = f"transform_v{config.transform_version}"
        f = globals().get(name)
        print(f"... Transform Info - {name}")
        return f(config)

    except TypeError:
        raise NotImplementedError("try another transform version ...")