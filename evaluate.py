import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import cv2
from model import unet 

def compute_dice_coefficient(y_true, y_pred, threshold=0.5):
    """Calculates the Dice Coefficient for segmentation evaluation."""
    y_true_f = np.ravel(y_true) 
    y_pred_f = np.ravel(y_pred > threshold)  
    intersection = np.sum(y_true_f * y_pred_f) 
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-6)  

def evaluate_model(model_path, X_val, Y_val, model=None):
    """Evaluates the trained U-Net model on validation data."""
    if model is None:
        print("Loading model...")
        model = unet(pretrained_weights=model_path)  # Load the model
        print("Model loaded successfully!")

    print("Predicting on validation data...")
    Y_pred_noth = model.predict(X_val)  
    Y_val_flat=[]
    Y_pred_flat = []
    # Calculate the ROC curve
   
    Y_val_flat= Y_val.flatten() 
    Y_pred_flat=Y_pred_noth.flatten()
    fpr, tpr, thresholds = roc_curve(Y_val_flat, Y_pred_flat, pos_label=1, drop_intermediate= True)  
    roc_auc = auc(fpr, tpr)  
    print("calculating dice coff")

    # Calculate the Dice scores for each threshold
    list =np.arange(0, 1, 0.005)
    dice_scores = [compute_dice_coefficient(Y_val_flat, Y_pred_flat, threshold=t) for t in list]

    # Find the best threshold based on the Dice score
    best_threshold_index = np.argmax(dice_scores) 
    best_threshold = list[best_threshold_index]  
    best_dice_score = dice_scores[best_threshold_index]

    print(f"Best Threshold: {best_threshold:.4f}, Dice Coefficient: {best_dice_score:.4f}")

 
    Y_pred = (Y_pred_noth > best_threshold).astype(np.uint8) 

    test_image = (Y_pred[0] * 255).astype(np.uint8)  
    

    # Plot and save the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--') 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])  
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic') 
    plt.legend(loc="lower right") 
    plt.savefig(f"roc_curve_{os.path.basename(model_path)}.png") 
    plt.close() 
    cv2.imwrite(f"image_test_{os.path.basename(model_path)}.jpg", test_image) 