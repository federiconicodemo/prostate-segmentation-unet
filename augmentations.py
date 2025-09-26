import cv2
import numpy as np
import random

def default_operation(image, mask):
    return image, mask

def random_rotation(image, mask, angle_range=(-15, 15)):
    angle = random.uniform(*angle_range)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    rotated_mask = cv2.warpAffine(mask, rot_matrix, (mask.shape[1], mask.shape[0]))

    return rotated_image, rotated_mask


def contrast_CLAHE(img, mask, clipLimit_range=(1, 5), tileGridSize_range=(4, 8)):
    clipLimit = random.uniform(*clipLimit_range)
    tile_size = random.randint(tileGridSize_range[0], tileGridSize_range[1])
    eq_hist = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tile_size, tile_size))
    aug_img = eq_hist.apply(img)
    return aug_img, mask


def gamma_correction(img, mask, gamma_range=(0.8, 1.2)):
    gamma = random.uniform(*gamma_range)
    inv_gamma = 1.0 / gamma

    lookUpTable = (np.linspace(0, 255, 256, dtype=np.float32) / 255.0) ** inv_gamma * 255
    lookUpTable = np.clip(lookUpTable, 0, 255).astype(np.uint8).reshape((256, 1))

    corrected_img = cv2.LUT(img.astype(np.uint8), lookUpTable)

    return corrected_img, mask


def gaussian_blur(image, mask, kernel_size=(3, 3)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image, mask


def add_gaussian_noise(image, mask, mean=0, stddev=5):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.int16)
    image = image.astype(np.int16)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image, mask


def flip_image(image, mask):
    flip_code = random.choice([-1, 0, 1])
    flipped_image = cv2.flip(image, flip_code)
    flipped_mask = cv2.flip(mask, flip_code)
    return flipped_image, flipped_mask


def apply_translation(image, mask, x_range=(-10, 10), y_range=(-10, 10)):
    tx = random.randint(*x_range)
    ty = random.randint(*y_range)
    trans_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, trans_matrix, (image.shape[1], image.shape[0]))
    translated_mask = cv2.warpAffine(mask, trans_matrix, (mask.shape[1], mask.shape[0]))
    return translated_image, translated_mask


def apply_scaling(image, mask, scale_range=(0.9, 1.1)):
    scale = random.uniform(*scale_range)
    scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return scaled_image, scaled_mask
