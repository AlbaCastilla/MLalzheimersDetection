import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

train_path = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/train"
test_path = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/test"

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