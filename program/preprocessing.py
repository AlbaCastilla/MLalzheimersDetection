import os
# os: allows interaction with the operating system, such as accessing files and directorie
import numpy as np
# NumPy: used for numerical operations and handling arrays and matrices
import matplotlib.pyplot as plt
# Matplotlib: used to create graphs and visualizations (e.g., class distribution, loss curves)
from collections import Counter
# Counter: counts occurrences of elements (useful for class distribution analysis)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ImageDataGenerator: loads, preprocesses, augments, and labels image data automatically
from tensorflow.keras.models import Sequential
# Sequential: allows building a neural network layer by layer in order
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Conv2D: extracts features from images using convolution filters
# MaxPooling2D: reduces image size while keeping important features
# Flatten: converts 2D feature maps into a 1D vector
# Dense: fully connected layer used for classification
# Dropout: randomly disables neurons to reduce overfitting
from sklearn.metrics import classification_report, confusion_matrix
# classification_report: computes precision, recall, and F1-score
# confusion_matrix: shows correct vs incorrect predictions per class
from sklearn.utils.class_weight import compute_class_weight
# compute_class_weight: handles class imbalance by giving more importance to minority classes
from PIL import Image
# PIL: used for image processing tasks such as opening, manipulating, and saving images



#dataset paths
train_path = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/train"
test_path = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/test"

# Checks a dataset of images in class folders for:

# 1. Allowed file types (.jpg, .jpeg, .png)
# 2. Openable (not corrupted) images
# 3. Correct image size (128x128)
# 4. Correct color mode ('L' for grayscale)
# 5. Filenames starting with class folder name (without spaces)

# Reports all issues found, removed corrupted files, and shows all color modes detected.

IMG_SIZE = (128, 128)  # expected image size
COLOR_MODE = "L"        # expected color mode
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"]  # allowed file types

# ----------------------------
# Dataset Validation & Normalization
# ----------------------------
def validate_and_normalize_dataset(dataset_path):
    """
    Validates images in the dataset and normalizes pixel values to 0-1.
    Returns a dictionary with normalized images by class.
    """
    issues = []  # store only actual issues
    removed_files = []
    color_modes_found = set()  # track all color modes in dataset
    dataset_normalized = {}    # store normalized images: {class_name: [image arrays]}

    for class_folder in os.listdir(dataset_path):
        class_folder_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_folder_path):
            continue

        # Initialize list for this class
        dataset_normalized[class_folder] = []

        # Prepare the expected prefix for filenames: remove spaces
        expected_prefix = class_folder.replace(" ", "")

        for filename in os.listdir(class_folder_path):
            file_path = os.path.join(class_folder_path, filename)

            # Check file extension
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                issues.append(f"File '{file_path}' has unusual extension '{ext}'")
                continue  # skip further checks for this file

            # Try opening image
            try:
                img = Image.open(file_path)
                color_modes_found.add(img.mode)

                # Convert to expected color mode if needed
                if img.mode != COLOR_MODE:
                    img = img.convert(COLOR_MODE)

                # Resize to expected size if needed
                if img.size != IMG_SIZE:
                    img = img.resize(IMG_SIZE)

                # Convert to NumPy array and normalize pixels to 0-1
                img_array = np.array(img) / 255.0

                # Add normalized image to dataset
                dataset_normalized[class_folder].append(img_array)

            except Exception as e:
                issues.append(f"Cannot open image '{file_path}': {e}")
                removed_files.append(file_path)
                os.remove(file_path)  # remove corrupted file
                continue

            # Check image size
            if img.size != IMG_SIZE:
                issues.append(f"Image '{file_path}' size {img.size} does not match expected {IMG_SIZE}")

            # Check color mode
            if img.mode != COLOR_MODE:
                issues.append(f"Image '{file_path}' mode '{img.mode}' does not match expected '{COLOR_MODE}'")

            # Check if filename starts with class folder name (without spaces)
            file_base = os.path.splitext(filename)[0]  # remove extension
            if not file_base.startswith(expected_prefix):
                issues.append(f"Image '{file_path}' filename does not start with class folder name '{expected_prefix}'")

    # Report issues
    if issues:
        print("\nDataset Issues Found:")
        for issue in issues:
            print("-", issue)
    else:
        print("Dataset validation passed. No issues found!")

    if removed_files:
        print(f"\nRemoved {len(removed_files)} corrupted/invalid files.")

    # Show color modes found
    print("\nColor modes found in dataset:", ", ".join(sorted(color_modes_found)))

    # Return normalized dataset for use in training
    return dataset_normalized

# ----------------------------
# RUN VALIDATION AND NORMALIZATION
# ----------------------------
print("Validating and normalizing TRAIN dataset...")
normalized_train = validate_and_normalize_dataset(train_path)

# Example: check number of images per class
for class_name, images in normalized_train.items():
    print(f"Class '{class_name}' has {len(images)} images.")