import argparse
import numpy as np
import tensorflow as tf
import cv2 as cv
import random
import datetime
from train import train_model, dice_loss, dice_coefficient
from evaluate import evaluate_model
from data_loader import load_metadata, load_dicom_from_metadata, load_nrrd_masks
from preprocess import preprocess_dataset,normalize255


 
def main():
    tf.config.optimizer.set_jit(True)
    tf.config.experimental.enable_mlir_graph_optimization = True
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    parser = argparse.ArgumentParser(description="Prostate Segmentation Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the U-Net model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model")
    parser.add_argument("--evaluatePath", type=str, help="Path for evaluation")     
    args = parser.parse_args()

    metadata_path = "../prostate segmentation dataset/NCI-ISBI prostate segmentation dataset/manifest-ZqaK9xEy8795217829022780222/metadata.csv" 
    base_dicom_path = "../prostate segmentation dataset/NCI-ISBI prostate segmentation dataset/manifest-ZqaK9xEy8795217829022780222/"
    nrrd_mask_path = "../prostate segmentation dataset/prostate annotations/"
    model_path = "prostate_segmentation_model.h5"
    
    print("Loading dataset...")
    metadata_df = load_metadata(metadata_path)
    dicom_volumes_siemens,dicom_volumes_philips = load_dicom_from_metadata(metadata_df, base_dicom_path)
    mask_arrays_siemens,mask_arrays_philips = load_nrrd_masks(nrrd_mask_path)
    
    print("Preprocessing dataset...")
    
    processed_images_siemens, processed_masks_siemens = preprocess_dataset(dicom_volumes_siemens, mask_arrays_siemens,target_size=(256,256))
    processed_images_philips, processed_masks_philips = preprocess_dataset(dicom_volumes_philips,mask_arrays_philips,target_size=(256,256))
    
    print("Extracting individual slices...")
    X_siemens = []
    Y_siemens = []
    X_philips = []
    Y_philips = []
    X_validate_si = []
    Y_validate_si = []
    X_validate_pi = []
    Y_validate_pi = []
    for subject_id in processed_images_siemens.keys():
        if random.randint(1,10) ==1:
            
            for slice_idx in range(len(processed_images_siemens[subject_id])):  # Iterate over slices
                if processed_masks_siemens[subject_id][slice_idx].any()>0:
                    X_validate_si.append(processed_images_siemens[subject_id][slice_idx])
                    Y_validate_si.append(processed_masks_siemens[subject_id][slice_idx]) 
        else:
            for slice_idx in range(len(processed_images_siemens[subject_id])):  # Iterate over slices
                if processed_masks_siemens[subject_id][slice_idx].any()>0:
                    X_siemens.append(processed_images_siemens[subject_id][slice_idx])
                    Y_siemens.append(processed_masks_siemens[subject_id][slice_idx])
    for subject_id in processed_images_philips.keys():
        if   random.randint(1,10) ==1:
            for slice_idx in range(len(processed_images_philips[subject_id])):  # Iterate over slices
                if processed_masks_philips[subject_id][slice_idx].any()>0:
                    X_validate_pi.append(processed_images_philips[subject_id][slice_idx])
                    Y_validate_pi.append(processed_masks_philips[subject_id][slice_idx]) 
        else:
            for slice_idx in range(len(processed_images_philips[subject_id])):  
                if processed_masks_philips[subject_id][slice_idx].any()>0:
                    X_philips.append(processed_images_philips[subject_id][slice_idx])
                    Y_philips.append(processed_masks_philips[subject_id][slice_idx])

    X_siemens = np.expand_dims(np.array(X_siemens), axis=-1)
    Y_siemens = np.expand_dims(np.array(Y_siemens), axis=-1)
    X_philips = np.expand_dims(np.array(X_philips), axis=-1)
    Y_philips= np.expand_dims(np.array(Y_philips), axis=-1)
    X_validate_pi = np.expand_dims(np.array(X_validate_pi), axis=-1)
    Y_validate_pi = np.expand_dims(np.array(Y_validate_pi), axis=-1)
    X_validate_si = np.expand_dims(np.array(X_validate_si), axis=-1)
    Y_validate_si= np.expand_dims(np.array(Y_validate_si), axis=-1)
    print(np.shape(X_siemens))
    print(np.shape(Y_siemens))
    print(np.shape(X_philips))
    print(np.shape(Y_philips))
    print(np.shape(X_validate_pi))
    print(np.shape(Y_validate_pi))
    print(np.shape(X_validate_si))
    print(np.shape(Y_validate_si))
    Y_siemens= np.where(Y_siemens > 0, 1, 0).astype(np.uint8)
    Y_philips= np.where(Y_philips > 0, 1, 0).astype(np.uint8)
    Y_validate_si= np.where(Y_validate_si > 0, 1, 0).astype(np.uint8)
    Y_validate_pi= np.where(Y_validate_pi > 0, 1, 0).astype(np.uint8)
    epochs = 100

    print(Y_philips.sum())
    print(np.where(Y_philips == 0 , 1,0).sum())
    

    if args.train and args.evaluate:

            print("Starting model training...")
            model_philips = train_model(X_philips,X_validate_pi,Y_philips, Y_validate_pi, epochs=epochs,name=f"model_philips_{datetime.datetime.now().timestamp()}.h5", alpha=0.5)

            #model_siemens = train_model(X_siemens,X_validate_si, Y_siemens, Y_validate_si, epochs=epochs,name=f"model_siemens_{datetime.datetime.now().timestamp()}.h5",alpha=0.65)
            
            print("Starting model evaluation...")
            print("siemens model")
            #evaluate_model(model_path,X_philips,Y_philips,model=model_siemens)
            print("philips model")
            #evaluate_model(model_path,X_siemens,Y_siemens,model=model_philips)

    else:
        if args.evaluatePath:
            print("Starting model evaluation...")

            #evaluate_model(args.evaluatePath,X_philips,Y_philips)
            #evaluate_model(args.evaluatePath, X_siemens, Y_siemens)

if __name__ == "__main__":
    main()
