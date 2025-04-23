import joblib
import numpy as np
from pathlib import Path

def load_model(model_path, scaler_path):
    """
    Load the trained model and scaler from disk.
    
    Args:
        model_path (str): Path to the saved model file
        scaler_path (str): Path to the saved scaler file
        
    Returns:
        tuple: (model, scaler) The loaded model and scaler objects
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict_pothole(model, scaler, features):
    """
    Make a prediction using the loaded model.
    
    Args:
        model: The loaded model object
        scaler: The loaded scaler object
        features (numpy.ndarray): Feature vector to predict on
        
    Returns:
        tuple: (is_pothole, confidence) Boolean prediction and confidence score
    """
    # Reshape features if needed
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    confidence = model.predict_proba(scaled_features)[0][1]  # Probability of pothole class
    
    return bool(prediction), float(confidence) 