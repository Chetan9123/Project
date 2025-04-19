import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import cv2

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        return img / 255.0
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def create_dataset():
    """Create training and testing datasets"""
    # Define paths
    data_dir = Path("data")
    train_pothole_dir = data_dir / "potholes"
    train_normal_dir = data_dir / "non_potholes"
    test_dir = data_dir / "test"
    
    # Load training images
    train_pothole_images = list(train_pothole_dir.glob("*.jpg"))
    train_normal_images = list(train_normal_dir.glob("*.jpg"))
    
    # Create training dataset
    X_train = []
    y_train = []
    
    # Load pothole images
    print(f"Processing {len(train_pothole_images)} pothole images...")
    for img_path in train_pothole_images:
        img = load_and_preprocess_image(img_path)
        if img is not None:
            X_train.append(img)
            y_train.append(1)  # 1 for pothole
    
    # Load normal road images
    print(f"Processing {len(train_normal_images)} normal road images...")
    for img_path in train_normal_images:
        img = load_and_preprocess_image(img_path)
        if img is not None:
            X_train.append(img)
            y_train.append(0)  # 0 for normal road
    
    if not X_train:
        raise ValueError("No valid images found in the training set!")
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Final training set size: {len(X_train)} images")
    print(f"Pothole images: {sum(y_train)}")
    print(f"Normal road images: {len(y_train) - sum(y_train)}")
    
    # Load test images
    test_images = list(test_dir.glob("*.jpg"))
    X_test = []
    test_image_paths = []
    
    print(f"Processing {len(test_images)} test images...")
    for img_path in test_images:
        img = load_and_preprocess_image(img_path)
        if img is not None:
            X_test.append(img)
            test_image_paths.append(img_path)
    
    if not X_test:
        raise ValueError("No valid images found in the test set!")
    
    X_test = np.array(X_test)
    print(f"Final test set size: {len(X_test)} images")
    
    return X_train, y_train, X_test, test_image_paths

def create_model():
    """Create and compile the CNN model"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model():
    """Main training function"""
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, test_image_paths = create_dataset()
    
    print("Creating and compiling model...")
    model = create_model()
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    print("Saving model...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / "pothole_model.h5")
    
    print("Making predictions on test set...")
    predictions = model.predict(X_test)
    
    # Save test predictions
    with open("test_predictions.txt", "w") as f:
        f.write("Image,Prediction\n")
        for img_path, pred in zip(test_image_paths, predictions):
            f.write(f"{img_path.name},{pred[0]:.4f}\n")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train_model()