import joblib
import numpy as np
from pathlib import Path
import warnings

def load_model(model_path, scaler_path):
    """
    Load the trained model and scaler
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        warnings.warn(
            f"Model or scaler files not found at {model_path} or {scaler_path}. "
            "Using dummy model for demonstration."
        )
        return DummyModel(), DummyScaler()

def predict_pothole(model, scaler, features):
    """
    Make prediction using the loaded model
    """
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    confidence = model.predict_proba(features_scaled)[0].max()
    
    return bool(prediction), float(confidence)

class DummyModel:
    """Dummy model for demonstration purposes"""
    def predict(self, X):
        return np.random.choice([0, 1], size=len(X))
    
    def predict_proba(self, X):
        probs = np.random.uniform(0.6, 0.9, size=(len(X), 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

class DummyScaler:
    """Dummy scaler for demonstration purposes"""
    def transform(self, X):
        return X 