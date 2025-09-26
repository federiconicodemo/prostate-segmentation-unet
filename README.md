# Prostate Gland Segmentation with U-Net in TensorFlow

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **Project Origin Note**
> This project was developed as a final assignment for a university course. The goal was to build a complete deep learning pipeline for a real-world medical imaging task.

This repository contains a complete pipeline for training and evaluating a U-Net model for segmenting the prostate gland from DICOM medical images. The project is implemented in TensorFlow/Keras and includes data loading, preprocessing, advanced data augmentation, model training, and performance evaluation.

## Key Features

-   **End-to-End Pipeline:** From raw DICOM files to a trained segmentation model and performance metrics.
-   **U-Net Architecture:** A standard, powerful deep learning model for biomedical image segmentation.
-   **DICOM and NRRD Support:** Handles DICOM series for input images and NRRD files for segmentation masks.
-   **Advanced Augmentation:** Includes standard augmentations (rotation, flipping, noise) and more complex strategies like **AugMix** to improve model robustness.
-   **Custom Loss Function:** Utilizes a combined loss of **Weighted Binary Cross-Entropy** and **IoU (Intersection over Union) Loss** to handle class imbalance and improve segmentation accuracy.
-   **Comprehensive Evaluation:** Automatically calculates the Dice Coefficient, plots the ROC curve, and determines the optimal prediction threshold on a validation set.
-   **Mixed-Precision Training:** Uses `mixed_float16` to accelerate training and reduce memory usage on compatible GPUs.

## Project Structure

```
.
├── main.py               # Main script to run training and evaluation
├── train.py              # Contains the training loop, loss functions, and metrics
├── evaluate.py           # Contains model evaluation logic and plotting
├── model.py              # Defines the U-Net architecture
├── data_loader.py        # Scripts for loading DICOM series and NRRD masks
├── preprocess.py         # Image and mask preprocessing (normalization, padding, resizing)
├── augmentations.py      # Library of individual augmentation functions
├── big_aug.py            # Applies a random sequence of augmentations
├── aug_mix.py            # Implements the AugMix data augmentation strategy
└── README.md             # This file

```
## Setup and Installation

### 1. Prerequisites
-   **Anaconda or Miniconda:** You must have Conda installed. If not, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [Anaconda](https://www.anaconda.com/products/distribution).
-   **NVIDIA GPU:** A GPU with CUDA support is highly recommended for reasonable training times. The Conda environment will handle the CUDA toolkit installation for TensorFlow.

### 2. Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/federiconicodemo/prostate-segmentation-unet.git
    cd prostate-segmentation-unet
    ```

2.  **Create the `environment.yml` File**
    Create a new file named `environment.yml` in the root of the project directory and paste the following content into it:
    ```yaml
    name: prostate-seg # The name for your conda environment

    channels:
      - conda-forge
      - defaults

    dependencies:
      - python=3.9
      - tensorflow
      - numpy
      - pandas
      - simpleitk
      - pynrrd
      - opencv
      - scipy
      - scikit-learn
      - matplotlib
    ```

3.  **Create and Activate the Conda Environment**
    Open your terminal (or Anaconda Prompt on Windows) in the project directory and run the following command to create the environment:
    ```bash
    conda env create -f environment.yml
    ```
    Once the installation is complete, activate the new environment:
    ```bash
    conda activate prostate-seg
    ```
    Your terminal prompt should now be prefixed with `(prostate-seg)`, indicating the environment is active.

### 3. Dataset
This project uses the **NCI-ISBI 2013 Challenge - Automated Segmentation of Prostate Structures** dataset. You need to download it and arrange the files as follows.

The code expects the following directory structure relative to the project root:
```
../prostate segmentation dataset/
├── NCI-ISBI prostate segmentation dataset/
│   └── manifest-ZqaK9xEy8795217829022780222/
│       ├── metadata.csv
│       └── NCI-ISBI-Challenge-Training-Data/
│           └── ... (all the DICOM patient folders)
│
└── prostate annotations/
    └── ... (all the .nrrd mask files)
```
*Note: The hardcoded paths in `main.py` point to this structure. You may need to adjust them if your data is located elsewhere.*

## Usage

**Important:** Before running any command, ensure your Conda environment is activated (`conda activate prostate-seg`).

### Training a New Model

To train the model from scratch, run the following command. This will train, evaluate, and save the model weights (`.h5` file) with a timestamp.
```bash
python main.py --train --evaluate
```

### Evaluating an Existing Model

To evaluate a pre-trained model on the validation set, use the `--evaluate` and `--evaluatePath` flags.
```bash
python main.py --evaluate --evaluatePath path/to/your/model.h5
```
Example:
```bash
python main.py --evaluate --evaluatePath model_philips_1678886400.h5
```

## Technical Details

-   **Model:** A standard U-Net implemented in TensorFlow/Keras.
-   **Loss Function:** `combined_loss = alpha * weighted_binary_crossentropy + (1 - alpha) * iou_loss`.
-   **Optimizer:** Adam with a learning rate of `1e-5`.
-   **Metrics:** The primary performance metric is the **Dice Coefficient**.
