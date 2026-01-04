# lets you work with files and folders in your computer
import os

# used for math stuff and arrays, makes calculations faster
import numpy as np

# used to open and work with images
from PIL import Image

# helps split the data into training and testing sets
from sklearn.model_selection import train_test_split

# used to balance the classes when some appear more than others
from sklearn.utils.class_weight import compute_class_weight

# converts labels into categorical format (like 0 -> [1,0,0])
from tensorflow.keras.utils import to_categorical

# used to create the neural network model step by step
from tensorflow.keras.models import Sequential

# layers for the CNN: convolutions, pooling, flattening, etc
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# used to check how well the model performs (metrics and errors)
from sklearn.metrics import classification_report, confusion_matrix



# 1 SETTINGS

#paths to dataset folders
train_path = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/train"
test_path = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/test"

IMG_SIZE = (128, 128)   # expected image size
COLOR_MODE = "L"         # grayscale (found in 1 image properties)
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"]  # allowed file types althpugh i think they are all jpg


# 2️ DATASET VALIDATION & NORMALIZATION

def validate_and_normalize_dataset(dataset_path):

    # Validates images in the dataset and normalizes pixel values to 0-1
    # Returns a dictionary with normalized images by class

    issues = []
    removed_files = []
    color_modes_found = set()
    dataset_normalized = {}

    for class_folder in os.listdir(dataset_path):
        class_folder_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_folder_path):
            continue

        dataset_normalized[class_folder] = []

        for filename in os.listdir(class_folder_path):
            file_path = os.path.join(class_folder_path, filename)

            # Check extension
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                issues.append(f"File '{file_path}' has unusual extension '{ext}'")
                continue

            # Open image
            try:
                img = Image.open(file_path)
                color_modes_found.add(img.mode)

                # Convert to expected color mode
                if img.mode != COLOR_MODE:
                    img = img.convert(COLOR_MODE)

                # Resize to expected size
                if img.size != IMG_SIZE:
                    img = img.resize(IMG_SIZE)

                # Normalize pixels to 0-1
                img_array = np.array(img) / 255.0

                # Store normalized image
                dataset_normalized[class_folder].append(img_array)

            except Exception as e:
                issues.append(f"Cannot open image '{file_path}': {e}")
                removed_files.append(file_path)
                os.remove(file_path)
                continue

    # Report
    if issues:
        print("\nDataset Issues Found:")
        for issue in issues:
            print("-", issue)
    else:
        print(" Dataset validation passed. No issues found!")

    if removed_files:
        print(f"\nRemoved {len(removed_files)} corrupted/invalid files.")

    print("\nColor modes found in dataset:", ", ".join(sorted(color_modes_found)))

    return dataset_normalized



# 3️ DATA PREPARATION & SPLITTING

def prepare_data(normalized_train, normalized_test, val_ratio=0.15):
    
    #Prepares the dataset:
    # Converts dictionary to NumPy arrays
    # Reshapes images
    # One-hot encodes labels
    # Splits train into train/validation
    # Keeps test as-is
    # Computes class weights based on training data
    
    #  Convert train dict to arrays 
    X_train, y_train = [], []
    class_names = list(normalized_train.keys())
    class_to_index = {name: i for i, name in enumerate(class_names)}

    for class_name, images in normalized_train.items():
        for img_array in images:
            X_train.append(img_array)
            y_train.append(class_to_index[class_name])

    X_train = np.array(X_train).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
    y_train = to_categorical(np.array(y_train), num_classes=len(class_names))

    #  Split train into train/validation 
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=42,
        stratify=y_train
    )

    # Convert test dict to arrays
    X_test, y_test = [], []
    for class_name, images in normalized_test.items():
        for img_array in images:
            X_test.append(img_array)
            y_test.append(class_to_index[class_name])

    X_test = np.array(X_test).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
    y_test = to_categorical(np.array(y_test), num_classes=len(class_names))

    # Compute class weights based on training set
    y_labels_train = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        'balanced', classes=np.arange(len(class_names)), y=y_labels_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights_dict, class_names


# 4 CNN MODEL CREATION

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 5️ RUN PIPELINE

print("Validating and normalizing TRAIN dataset...")
normalized_train = validate_and_normalize_dataset(train_path)

print("\nValidating and normalizing TEST dataset...")
normalized_test = validate_and_normalize_dataset(test_path)

print("\nPreparing data (train/validation/test)...")
X_train, X_val, X_test, y_train, y_val, y_test, class_weights, class_names = prepare_data(
    normalized_train, normalized_test, val_ratio=0.15
)

for class_name in class_names:
    print(f"Class '{class_name}' - train: {len(normalized_train[class_name])}, test: {len(normalized_test[class_name])}")

print("\nBuilding CNN model...")
model = build_cnn_model(input_shape=(128,128,1), num_classes=len(class_names))
model.summary()

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weights
)

print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc*100:.2f}%")


# 6️ TEST SET EVALUATION ( pwer class)

print("\nEvaluating per-class performance on test set...")

# Predict on test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)


# 7️ TRY IMAGE PREDICTION


def predict_single_image(image_path, model, class_names):
    #Took 2 images out of 2 train folders, loads an img, applies the same preprocessing as training, and predicts the Alzheimer class.

    # Load image
    img = Image.open(image_path)

    # Convert to grayscale if needed
    if img.mode != COLOR_MODE:
        img = img.convert(COLOR_MODE)

    # Resize
    img = img.resize(IMG_SIZE)

    # Normalize
    img_array = np.array(img) / 255.0

    # Reshape for model (1, 128, 128, 1)
    img_array = img_array.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

    # Predict
    prediction_probs = model.predict(img_array)
    predicted_class_index = np.argmax(prediction_probs)
    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction_probs[0][predicted_class_index]

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Predicted class: {predicted_class_name}")
    print(f"Confidence: {confidence * 100:.2f}%")

    return predicted_class_name, confidence


# Paths to images to test prediction
image_1 = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/test/Test1.jpg"
image_2 = "dataset/Alzheimer's Disease Multiclass Images Dataset/Combined Dataset/test/Test2.jpg"

# Run predictions 
predict_single_image(image_1, model, class_names)
predict_single_image(image_2, model, class_names)