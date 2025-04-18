import os
import pandas as pd
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense  
import joblib

# Load CSV File
csv_file = "C:/Users/Admin/Desktop/hiproject/Project/dataa.csv"
data = pd.read_csv(csv_file)

print("Dataset Sample:")
print(data.head())

# Convert Labels to Numeric (1 = Pothole, 0 = Normal)
data["Label"] = data["Label"].map({"pothole": 1, "normal": 0})

# Ensure Labels are float (needed for TensorFlow)
y = data["Label"].astype(float).values

# Extract HOG Features
def extract_hog_features(image_path):
    """ Load image, convert to grayscale, resize, and extract HOG features """
    if not os.path.exists(image_path):  # Check if file exists
        print(f"⚠️ Missing file: {image_path}")
        return None
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if image is None:
        print(f"⚠️ Error loading image: {image_path}")
        return None  # Skip processing this image

    image = cv2.resize(image, (64, 64))  # Resize safely
    features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return features

# Prepare dataset (Extract features from images)
X, y_filtered = [], []
for index, row in data.iterrows():
    image_path = row["image_path"]  # Ensure using "Path" instead of "image_path"
    feature = extract_hog_features(image_path)
    
    if feature is not None:
        X.append(feature)
        y_filtered.append(y[index])  # Only keep labels for valid images

# Convert to numpy arrays
X = np.array(X)
y_filtered = np.array(y_filtered)

# Standardize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler for future use (IMPORTANT)
joblib.dump(scaler, "scaler.pkl")

# Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_filtered, test_size=0.2, random_state=42)

# Define Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save("pothole_detector.keras")  # Save in the recommended format

# Save Model
model.save("pothole_detector.keras")  # Save in the recommended format
model.save("pothole-detection.h5")    # Save in HDF5 format

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")


