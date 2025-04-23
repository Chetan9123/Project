import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import cv2
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.image_processing import extract_features

def train_model():
    # Load CSV File
    csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "new_dataset.csv")
    data = pd.read_csv(csv_file)

    print("Dataset Sample:")
    print(data.head())

    # Convert Labels to Numeric (1 = Pothole, 0 = Normal)
    data["Label"] = data["Label"].map({"pothole": 1, "normal": 0})
    y = data["Label"].astype(float).values

    # Prepare dataset with error handling
    X, y_filtered = [], []
    for index, row in data.iterrows():
        image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), row["image_path"])
        try:
            image = cv2.imread(image_path)
            if image is not None:
                # Process image and extract features
                processed_img = cv2.resize(image, (224, 224))
                processed_img = processed_img.astype(np.float32) / 255.0
                feature = extract_features(processed_img)
                X.append(feature)
                y_filtered.append(y[index])
                print(f"Processed {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    if len(X) == 0:
        raise ValueError("No valid images were processed. Please check the image paths and formats.")

    # Convert to numpy arrays
    X = np.array(X)
    y_filtered = np.array(y_filtered)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of samples: {len(y_filtered)}")

    # Standardize Features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save the scaler
    scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_filtered, test_size=0.2, random_state=42)

    # Create and train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save Model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pothole_model.joblib")
    joblib.dump(model, model_path)

    # Evaluate Model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print("\nModel Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    train_model() 