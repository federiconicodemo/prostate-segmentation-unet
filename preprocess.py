import numpy as np
import cv2
from scipy.ndimage import zoom
from big_aug import *
from aug_mix import *
def normalize(img):
    img = img.astype(np.float32)
    normimg = img - img.min()
    normimg /= img.max()
    return normimg
def normalize255(img):
    img = img.astype(np.float32)
    normimg = img - img.min()
    normimg /= img.max()
    normimg *= 255
    return normimg.astype(np.uint8)



def add_padding(image, target_size=(256, 256), pad_value=0):
    """Add padding to the image to reach the target size."""
    height, width = image.shape[:2]
    target_height, target_width = target_size

    # Calculate padding
    pad_top = (target_height - height) // 2
    pad_bottom = target_height - height - pad_top
    pad_left = (target_width - width) // 2
    pad_right = target_width - width - pad_left

    # Apply padding
    if len(image.shape) == 3:  
        padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[pad_value]*image.shape[2])
    else:  
        padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)

    return padded_image

def preprocess_image(image, target_size=(256, 256)):
    
    """Apply normalization and padding to image."""
    image = normalize(image)

    height, width = image.shape[:2]
    if height > 256:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    else:
        image = add_padding(image, target_size, pad_value=0)
    return image

def preprocess_mask(mask, target_size=(256, 256)):
    """Apply padding to mask with zero padding."""
    height, width = mask.shape[:2]
    if height > 256:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)
    else:
        mask = add_padding(mask, target_size, pad_value=0)
    return mask.astype(np.uint8)



def preprocess_dataset(dicom_volumes, mask_arrays, target_size=(256, 256), target_depth=20):
    """Apply preprocessing steps to images and masks."""
    processed_images = {}
    processed_masks = {}

    for subject_id in dicom_volumes.keys():

        image = dicom_volumes[subject_id]
        mask = mask_arrays[subject_id] 
  
        
        processed_images[subject_id]=[]
        processed_masks[subject_id]=[]
        mask =np.transpose(mask, (2, 0,1))

        if mask.shape == image.shape:
            for slice_idx in range(image.shape[0]): 
                processed_images[subject_id].append(preprocess_image(image[slice_idx], target_size))
                processed_masks[subject_id].append(preprocess_mask(mask[slice_idx], target_size))
                '''for _ in range(5):
                    aug_img,aug_mask=aug_mix(normalize255(image[slice_idx]),mask[slice_idx])
                    processed_images[subject_id].append(preprocess_image(aug_img))
                    processed_masks[subject_id].append(preprocess_mask(aug_mask))'''
                    
        else:
            print(f"not same shape {subject_id}")
    
    return processed_images, processed_masks

