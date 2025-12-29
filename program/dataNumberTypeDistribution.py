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

# Function to summarize dataset (class distribution, number of imgs and data types)
def summarize_dataset(path):
    print(f"\nAnalyzing dataset in: {path}")
    class_counts = {}
    data_types = set()
    
    for class_name in os.listdir(path):
        if os.path.isdir(class_folder):
            files = os.listdir(class_folder)
            num_images = len(files)
            class_counts[class_name] = num_images
            
            # Identify data types
            for f in files:
                ext = f.split('.')[-1].lower()
                data_types.add(ext)
    
    # Print class counts
    total_images = sum(class_counts.values())
    print("Class distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / total_images) * 100
        print(f"{class_name}: {count} images ({percentage:.2f}%)")
    
    # Print data types
    print(f"Image file types in this dataset: {data_types}")
    
    # Plot distribution
    plt.figure(figsize=(8,5))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.title(f"Class Distribution in {os.path.basename(path)}")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.show()

# Summarize train and test sets
summarize_dataset(train_path)
#summarize_dataset(test_path)