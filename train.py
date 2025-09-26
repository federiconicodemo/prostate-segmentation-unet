import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from model import unet

class_weights = tf.constant([0.05, 0.95])

from tensorflow.keras.callbacks import Callback

class NaNChecker(Callback):
    def on_batch_end(self, batch, logs=None):
        if logs is not None and ('loss' in logs) and (tf.math.is_nan(logs['loss'])):
            print(f"⚠️ Loss is NaN at batch {batch}, stopping training!")
            self.model.stop_training = True
            
def weighted_binary_crossentropy(y_true, y_pred):

    bc = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bce = bc(y_true, y_pred)

    weight_vector = y_true * class_weights[1] + (1 - y_true) * class_weights[0]

    weighted_bce = weight_vector * bce
    return tf.reduce_mean(weighted_bce)

def iou_loss(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    iou = intersection / (union + K.epsilon())
    return 1 - K.mean(iou)

def create_combined_loss(alpha=0.65):
    def combined_loss(y_true, y_pred, from_logits=False):
        bce = weighted_binary_crossentropy(y_pred=y_pred, y_true=y_true)

        iou = iou_loss(y_true, y_pred)

        return alpha * bce + (1 - alpha) * iou

    return combined_loss
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Computes the Dice Coefficient for segmentation performance."""
    y_true_f = K.flatten(K.cast(y_true, 'float32'))  
    y_pred_f = K.flatten(K.cast(y_pred, 'float32')) 
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice Loss = 1 - Dice Coefficient (for training U-Net)."""
    return 1 - dice_coefficient(y_true, y_pred)

def train_model(X_train, X_val ,Y_train, Y_val ,batch_size=4, epochs=50,name="prostate_segmentation_model.h5", alpha = 0.65):
    """Trains the U-Net model on prostate segmentation data slice by slice."""
    early_stopping =keras.callbacks.EarlyStopping("val_loss",start_from_epoch=5,restore_best_weights=True,patience =5, verbose = 1)
    com= create_combined_loss(alpha=alpha)
    print("Building U-Net model...")
    model = unet(input_size=(256, 256, 1))
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=com, metrics=[dice_coefficient])
    
    print("Starting training...")
    history = model.fit(X_train, Y_train,validation_data =(X_val,Y_val),
                        batch_size=batch_size,
                        epochs=epochs,callbacks=[NaNChecker(), early_stopping])
    
    print("Training complete!")
    model.save(name)
    print(f"Model saved as {name}")

    return model