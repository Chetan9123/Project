from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import numpy as np
from flask import current_app
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_advanced_model(input_dim):
    """Create an advanced neural network model"""
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_ml_models():
    """Load machine learning models using config paths"""
    try:
        model = load_model(current_app.config['MODEL_PATH'])
        scaler = joblib.load(current_app.config['SCALER_PATH'])
        rf_model = joblib.load(current_app.config['RF_MODEL_PATH'])
    except:
        # If models don't exist, create new ones
        model = create_advanced_model(2048)  # Adjust input_dim based on feature size
        scaler = None
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    return model, scaler, rf_model

def ensemble_predict(image_features, model, scaler, rf_model):
    """Make prediction using ensemble of models"""
    # Scale features if scaler exists
    if scaler is not None:
        features = scaler.transform([image_features])
    else:
        features = [image_features]
    
    # Get predictions from both models
    nn_pred = model.predict(features)[0][0]
    rf_pred = rf_model.predict_proba(features)[0][1]
    
    # Ensemble prediction (weighted average)
    ensemble_pred = 0.6 * nn_pred + 0.4 * rf_pred
    
    # Determine final prediction and confidence
    is_pothole = ensemble_pred > 0.5
    confidence = float(ensemble_pred if is_pothole else 1 - ensemble_pred)
    
    return is_pothole, confidence

def evaluate_model(model, rf_model, X_test, y_test):
    """Evaluate model performance"""
    # Neural Network predictions
    nn_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Random Forest predictions
    rf_pred = rf_model.predict(X_test)
    
    # Ensemble predictions
    ensemble_pred = (0.6 * model.predict(X_test) + 0.4 * rf_model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
    
    # Calculate metrics for each model
    models = {
        'Neural Network': nn_pred,
        'Random Forest': rf_pred,
        'Ensemble': ensemble_pred
    }
    
    results = {}
    for name, predictions in models.items():
        results[name] = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions)
        }
    
    return results 