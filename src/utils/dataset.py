import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from .image_processing import extract_features, enhance_image
import joblib
from flask import current_app

def prepare_dataset(data_dir, test_size=0.2, random_state=42):
    """
    Prepare the dataset by loading images, extracting features, and splitting into train/test sets.
    
    Args:
        data_dir (str): Directory containing the dataset
        test_size (float): Proportion of dataset to include in the test split
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Initialize lists to store features and labels
    features = []
    labels = []
    
    # Process pothole images
    pothole_dir = os.path.join(data_dir, 'potholes')
    for img_name in os.listdir(pothole_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(pothole_dir, img_name)
            try:
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Enhance image
                enhanced = enhance_image(img)
                
                # Extract features
                img_features = extract_features(enhanced)
                
                features.append(img_features)
                labels.append(1)  # 1 for pothole
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    # Process non-pothole images
    non_pothole_dir = os.path.join(data_dir, 'non_potholes')
    for img_name in os.listdir(non_pothole_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(non_pothole_dir, img_name)
            try:
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Enhance image
                enhanced = enhance_image(img)
                
                # Extract features
                img_features = extract_features(enhanced)
                
                features.append(img_features)
                labels.append(0)  # 0 for non-pothole
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    scaler_path = os.path.join(current_app.config['BASE_DIR'], 'models', 'scaler.pkl')
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def load_and_prepare_single_image(image_path):
    """
    Load and prepare a single image for prediction.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        numpy.ndarray: Prepared features for the image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")
    
    # Enhance image
    enhanced = enhance_image(img)
    
    # Extract features
    features = extract_features(enhanced)
    
    # Load scaler
    scaler_path = os.path.join(current_app.config['BASE_DIR'], 'models', 'scaler.pkl')
    scaler = joblib.load(scaler_path)
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    return features_scaled

def get_dataset_statistics(data_dir):
    """
    Get statistics about the dataset.
    
    Args:
        data_dir (str): Directory containing the dataset
    
    Returns:
        dict: Dataset statistics
    """
    stats = {
        'total_images': 0,
        'pothole_images': 0,
        'non_pothole_images': 0,
        'image_sizes': [],
        'file_types': {}
    }
    
    # Count pothole images
    pothole_dir = os.path.join(data_dir, 'potholes')
    for img_name in os.listdir(pothole_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            stats['pothole_images'] += 1
            stats['total_images'] += 1
            
            # Get file extension
            ext = os.path.splitext(img_name)[1].lower()
            stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
            
            # Get image size
            img_path = os.path.join(pothole_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                stats['image_sizes'].append(img.shape)
    
    # Count non-pothole images
    non_pothole_dir = os.path.join(data_dir, 'non_potholes')
    for img_name in os.listdir(non_pothole_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            stats['non_pothole_images'] += 1
            stats['total_images'] += 1
            
            # Get file extension
            ext = os.path.splitext(img_name)[1].lower()
            stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
            
            # Get image size
            img_path = os.path.join(non_pothole_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                stats['image_sizes'].append(img.shape)
    
    # Calculate average image size
    if stats['image_sizes']:
        avg_size = np.mean(stats['image_sizes'], axis=0)
        stats['average_image_size'] = {
            'height': int(avg_size[0]),
            'width': int(avg_size[1]),
            'channels': int(avg_size[2])
        }
    
    return stats 