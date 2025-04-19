import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from utils.image_processing import extract_hog_features
import cv2

def train_model():
    # Load CSV File
    csv_file = "../dataa.csv"
    data = pd.read_csv(csv_file)

    print("Dataset Sample:")
    print(data.head())

    # Convert Labels to Numeric (1 = Pothole, 0 = Normal)
    data["Label"] = data["Label"].map({"pothole": 1, "normal": 0})
    y = data["Label"].astype(float).values

    # Prepare dataset with error handling
    X, y_filtered = [], []
    for index, row in data.iterrows():
        image_path = row["image_path"]
        image = cv2.imread(image_path)
        if image is not None:
            feature = extract_hog_features(image)
            X.append(feature)
            y_filtered.append(y[index])

    # Convert to numpy arrays
    X = np.array(X)
    y_filtered = np.array(y_filtered)

    # Standardize Features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, "../scaler.pkl")

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_filtered, test_size=0.2, random_state=42)

    # Define improved Neural Network Model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])

    # Compile Model
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # Train Model
    history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

    # Save Model
    model.save("../pothole_detector.keras")

    # Evaluate Model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Print training history
    print("\nTraining History:")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Best Training Accuracy: {max(history.history['accuracy']):.4f}")

if __name__ == '__main__':
    train_model() 