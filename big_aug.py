import random
from augmentations import *

def big_aug(image, mask):
    transformations = [
        random_rotation,
        contrast_CLAHE,
        gaussian_blur,
        add_gaussian_noise,
        flip_image,
        apply_translation,
        apply_scaling,
        gamma_correction
    ]

    n = random.randint(2, 5)
    for _ in range(n):
        transform = random.choice(transformations)
        image, mask = transform(image, mask)
    return image, mask