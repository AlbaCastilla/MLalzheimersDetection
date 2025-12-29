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

#dataset paths
train_path = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/train"
test_path = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/test"

# Function to count images per class
def count_images(path):
    print(f"Counting images in: {path}")
    for class_name in os.listdir(path):
        class_folder = os.path.join(path, class_name)
        if os.path.isdir(class_folder):
            num_images = len(os.listdir(class_folder))
            print(f"{class_name}: {num_images} images")
            
# Count images in train and test sets
count_images(train_path)
count_images(test_path)