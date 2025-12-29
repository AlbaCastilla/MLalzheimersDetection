import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


train_path = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/train"
test_path = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/test"

def summarize_dataset(path):
    print(f"\nAnalyzing dataset in: {path}")
    class_counts = {}
    data_types = set()
    
    for class_name in os.listdir(path):
        class_folder = os.path.join(path, class_name)
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