import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nrrd
import cv2
from glob import glob

def load_metadata(csv_path):
    """Loads the metadata CSV file into a Pandas DataFrame."""
    return pd.read_csv(csv_path)

def load_dicom_series(dicom_folder):
    """Loads a series of DICOM images into a 3D volume."""
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_series)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)  # Convert to numpy (Z, H, W)

def convert_path(path):
    path = path.replace('\\', '/')  # Replace backslashes with forward slashes

    return path

def load_dicom_from_metadata(metadata_df, base_path):
    """Loads DICOM images based on file locations specified in metadata."""
    dicom_volumes_siemens = {}
    dicom_volumes_philips = {}
    print(metadata_df['Series Description'])
    siemens_df = metadata_df[metadata_df['Series Description'] == 'SIEMENS']
    philips_df = metadata_df[metadata_df['Series Description'] == 'Philips Medical Systems'] 
    for _, row in siemens_df.iterrows():
        dicom_folder_siemens = os.path.join(base_path,convert_path( row['File Location']))
        
        if os.path.exists(dicom_folder_siemens):
            dicom_volumes_siemens[row['Data Description URI']] = load_dicom_series(dicom_folder_siemens)
        else:
            print(f"Warning: DICOM folder {dicom_folder_siemens} not found.")
    for _, row in philips_df.iterrows():
        dicom_folder_philips = os.path.join(base_path,convert_path( row['File Location']))
        
        if os.path.exists(dicom_folder_philips):
            dicom_volumes_philips[row['Data Description URI']] = load_dicom_series(dicom_folder_philips)
        else:
            print(f"Warning: DICOM folder {dicom_folder_philips} not found.")
    
    return dicom_volumes_siemens,dicom_volumes_philips

def load_nrrd_masks(nrrd_folder):
    """Loads all NRRD segmentation masks from a given directory."""
    masks_siemens = {}
    masks_philips = {}
    
    for nrrd_file in glob(os.path.join(nrrd_folder, "*.nrrd")):
        
        subject_id = os.path.basename(nrrd_file).split('.')[0]
        
        data, _ = nrrd.read(nrrd_file) 
        print(subject_id.startswith('Prostate3T'))
        if subject_id.startswith('Prostate3T'):
            masks_siemens[subject_id] = data
        else :
            masks_philips[subject_id] = data
    
    return masks_siemens,masks_philips

def preprocess_image(image, target_size=(256, 256)):
    """Apply normalization and resizing to image."""
    image = image.astype(np.float32)
    image -= np.min(image)
    image /= np.max(image)
    resized_slices = [cv2.resize(slice, target_size, interpolation=cv2.INTER_LINEAR) for slice in image]
    return np.stack(resized_slices, axis=0)

def preprocess_mask(mask, target_size=(256, 256)):
    """Resize mask and ensure binary values (0 or 1)."""
    resized_slices = [cv2.resize(slice, target_size, interpolation=cv2.INTER_NEAREST) for slice in mask]
    mask = np.stack(resized_slices, axis=0)
    mask[mask > 0] = 1  
    return mask.astype(np.uint8)

def preprocess_dataset(dicom_volumes, mask_arrays, target_size=(256, 256)):
    """Apply preprocessing steps to images and masks."""
    processed_images = {}
    processed_masks = {}
    
    for subject_id, image in dicom_volumes.items():
        mask = mask_arrays.get(subject_id, np.zeros_like(image))  
        processed_images[subject_id] = preprocess_image(image, target_size)
        processed_masks[subject_id] = preprocess_mask(mask, target_size)
    
    return processed_images, processed_masks


