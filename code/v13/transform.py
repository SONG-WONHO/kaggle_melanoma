from albumentations.pytorch import ToTensor
from albumentations import *
from torchvision import transforms, models
import cv2
import random

import os
import numpy as np
import albumentations
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations import functional as F


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
def transform_v99(config):
    """ tta transforms

        Args:
            config: CFG

        Returns: test_transforms
        """

    test_transforms = Compose([
        Flip(p=1),
        # RandomRotate90(p=1),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor()
    ], p=1)

    return test_transforms


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
    """ Flip, Rotate, HueSaturationValue

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        Flip(p=1),
        RandomRotate90(p=1),
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


def transform_v6(config):
    """ Flip, Rotate, RandomBrightnessContrast, Cutout

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        Flip(p=1),
        RandomRotate90(p=1),
        RandomBrightnessContrast(0.1, 0.1, p=1),
        Cutout(num_holes=4, max_h_size=4, max_w_size=4, p=0.5),
        Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.5),
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


def transform_v7(config):
    """ RandomResizedCrop, Flip, Rotate, RandomBrightnessContrast

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        RandomResizedCrop(224, 224, (0.7, 1), p=1),
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


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]


def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    return image - 127


def apply_op(image, op, severity):
    #   image = np.clip(image, 0, 255)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img)


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
    """
    ws = np.float32(
      np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image).astype(np.float32)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug
#         mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * image + m * mix
#     mixed = (1 - m) * normalize(image) + m * mix
    return mixed


class RandomAugMix(ImageOnlyTransform):

    def __init__(self, severity=3, width=3, depth=-1, alpha=1., always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def apply(self, image, **params):
        image = augment_and_mix(
            image,
            self.severity,
            self.width,
            self.depth,
            self.alpha
        )
        return image


def transform_v8(config):
    """ Flip, Rotate, RandomBrightnessContrast, Augmix

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        Flip(p=1),
        RandomRotate90(p=1),
        RandomBrightnessContrast(0.1, 0.1, p=1),
        RandomAugMix(severity=3, width=3, alpha=1., p=0.5),
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


def transform_v9(config):
    """ Flip, Rotate, RandomBrightnessContrast, Augmix

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        Flip(p=1),
        RandomRotate90(p=1),
        RandomBrightnessContrast(0.1, 0.1, p=1),
        RandomAugMix(severity=5, width=5, alpha=1., p=0.5),
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


def transform_v10(config):
    """ Flip, Rotate, RandomBrightnessContrast, Augmix

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = Compose([
        Flip(p=1),
        RandomRotate90(p=1),
        RandomBrightnessContrast(0.1, 0.1, p=1),
        RandomAugMix(severity=7, width=7, alpha=1., p=0.5),
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


### torch vision
class UniformAugment(object):
    def __init__(self, NumOps=2, fillcolor=(128, 128, 128)):
        self.NumOps = NumOps
        self.augs = {
            'shearX': [-0.3, 0.3],
            'shearY': [-0.3, 0.3],
            'translateX': [-0.45, 0.45],
            'translateY': [-0.45, 0.45],
            'rotate': [-30, 30],
            'autocontrast': [0, 0],
            'invert': [0, 0],
            'equalize': [0, 0],
            'solarize': [0, 256],
            'posterize': [4, 8],
            'contrast': [0.1, 1.9],
            'color': [0.1, 1.9],
            'brightness': [0.1, 1.9],
            'sharpness': [0.1, 1.9],
            'cutout': [0, 0.2]
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert('RGBA').rotate(magnitude)
            return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4), rot).convert(img.mode)

        def cutout(img, magnitude, fillcolor):
            img = img.copy()
            w, h = img.size
            v = w * magnitude
            x0 = np.random.uniform(w)
            y0 = np.random.uniform(h)
            x0 = int(max(0, x0 - v / 2.))
            y0 = int(max(0, y0 - v / 2.))
            x1 = min(w, x0 + v)
            y1 = min(h, y0 + v)
            xy = (x0, y0, x1, y1)
            ImageDraw.Draw(img).rectangle(xy, fillcolor)
            return img

        self.func = {
            'shearX': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0), Image.BICUBIC, fillcolor=fillcolor),
            'shearY': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0), Image.BICUBIC, fillcolor=fillcolor),
            'translateX': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0), fillcolor=fillcolor),
            'translateY': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude), fillcolor=fillcolor),
            'rotate': lambda img, magnitude: rotate_with_fill(img, magnitude),
            'color': lambda img, magnitude: ImageEnhance.Color(img).enhance(magnitude),
            'posterize': lambda img, magnitude: ImageOps.posterize(img, int(magnitude)),
            'solarize': lambda img, magnitude: ImageOps.solarize(img, int(magnitude)),
            'contrast': lambda img, magnitude: ImageEnhance.Contrast(img).enhance(magnitude),
            'sharpness': lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(magnitude),
            'brightness': lambda img, magnitude: ImageEnhance.Brightness(img).enhance(magnitude),
            'autocontrast': lambda img, magnitude: ImageOps.autocontrast(img),
            'equalize': lambda img, magnitude: ImageOps.equalize(img),
            'invert': lambda img, magnitude: ImageOps.invert(img),
            'cutout': lambda img, magnitude: cutout(img, magnitude, fillcolor=fillcolor)
        }

    def __call__(self, img):
        operations = random.sample(list(self.augs.items()), self.NumOps)
        for operation in operations:
            aug, range = operation
            magnitude = random.uniform(range[0], range[1])
            probability = random.random()
            if random.random() < probability:
                img = self.func[aug](img, magnitude)
        return img


def transform_v11(config):
    """ UniformAugment

    Args:
        config: CFG

    Returns: train_tranforms, test_transforms
    """
    train_transforms = transforms.Compose([
        UniformAugment(),
        transforms.RandomResizedCrop(size=config.image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

    return train_transforms, test_transforms


def get_transform(config):
    try:
        name = f"transform_v{config.transform_version}"
        f = globals().get(name)
        print(f"... Transform Info - {name}")
        return f(config)

    except TypeError:
        raise NotImplementedError("try another transform version ...")