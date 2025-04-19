import os
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.dataset import prepare_dataset, load_and_prepare_single_image
from ...config.config import config

def load_models():
    """Load the trained neural network and random forest models"""
    try:
        nn_model = load_model(config['default'].MODEL_PATH)
        rf_model = joblib.load(config['default'].RF_MODEL_PATH)
        scaler = joblib.load(config['default'].SCALER_PATH)
        return nn_model, rf_model, scaler
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None

def evaluate_models(X_test, y_test, nn_model, rf_model):
    """Evaluate both models and their ensemble"""
    # Neural Network predictions
    nn_pred_proba = nn_model.predict(X_test)
    nn_pred = (nn_pred_proba > 0.5).astype(int)
    
    # Random Forest predictions
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_pred_proba > 0.5).astype(int)
    
    # Ensemble predictions
    ensemble_pred_proba = (config['default'].NN_WEIGHT * nn_pred_proba.flatten() + 
                         config['default'].RF_WEIGHT * rf_pred_proba)
    ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
    
    # Print classification reports
    print("\nNeural Network Classification Report:")
    print(classification_report(y_test, nn_pred))
    
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_pred))
    
    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred))
    
    # Plot confusion matrices
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    sns.heatmap(confusion_matrix(y_test, nn_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Neural Network Confusion Matrix')
    
    plt.subplot(132)
    sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    
    plt.subplot(133)
    sns.heatmap(confusion_matrix(y_test, ensemble_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Ensemble Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png')
    plt.close()
    
    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_pred_proba)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
    ensemble_fpr, ensemble_tpr, _ = roc_curve(y_test, ensemble_pred_proba)
    
    nn_auc = auc(nn_fpr, nn_tpr)
    rf_auc = auc(rf_fpr, rf_tpr)
    ensemble_auc = auc(ensemble_fpr, ensemble_tpr)
    
    plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
    plt.plot(ensemble_fpr, ensemble_tpr, label=f'Ensemble (AUC = {ensemble_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig('results/roc_curves.png')
    plt.close()
    
    return {
        'nn_accuracy': np.mean(nn_pred == y_test),
        'rf_accuracy': np.mean(rf_pred == y_test),
        'ensemble_accuracy': np.mean(ensemble_pred == y_test),
        'nn_auc': nn_auc,
        'rf_auc': rf_auc,
        'ensemble_auc': ensemble_auc
    }

def test_single_image(image_path, nn_model, rf_model, scaler):
    """Test a single image with both models and return ensemble prediction"""
    try:
        # Load and prepare the image
        features = load_and_prepare_single_image(image_path, scaler)
        
        # Make predictions
        nn_pred = nn_model.predict(features.reshape(1, -1))[0][0]
        rf_pred = rf_model.predict_proba(features.reshape(1, -1))[0][1]
        
        # Ensemble prediction
        ensemble_pred = (config['default'].NN_WEIGHT * nn_pred + 
                        config['default'].RF_WEIGHT * rf_pred)
        
        # Determine final prediction
        is_pothole = ensemble_pred > 0.5
        confidence = float(ensemble_pred if is_pothole else 1 - ensemble_pred)
        
        return {
            'is_pothole': bool(is_pothole),
            'confidence': confidence,
            'nn_confidence': float(nn_pred),
            'rf_confidence': float(rf_pred)
        }
    except Exception as e:
        print(f"Error testing image: {str(e)}")
        return None

def main():
    # Load models
    nn_model, rf_model, scaler = load_models()
    if nn_model is None or rf_model is None or scaler is None:
        print("Failed to load models. Exiting.")
        return
    
    # Prepare test dataset
    X_train, X_test, y_train, y_test, _ = prepare_dataset('data')
    
    # Evaluate models
    results = evaluate_models(X_test, y_test, nn_model, rf_model)
    
    print("\nModel Evaluation Results:")
    print(f"Neural Network Accuracy: {results['nn_accuracy']:.4f}")
    print(f"Random Forest Accuracy: {results['rf_accuracy']:.4f}")
    print(f"Ensemble Accuracy: {results['ensemble_accuracy']:.4f}")
    print(f"Neural Network AUC: {results['nn_auc']:.4f}")
    print(f"Random Forest AUC: {results['rf_auc']:.4f}")
    print(f"Ensemble AUC: {results['ensemble_auc']:.4f}")
    
    # Test a single image
    test_image = 'data/test/test_image.jpg'
    if os.path.exists(test_image):
        result = test_single_image(test_image, nn_model, rf_model, scaler)
        if result:
            print("\nSingle Image Test Result:")
            print(f"Is Pothole: {result['is_pothole']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Neural Network Confidence: {result['nn_confidence']:.4f}")
            print(f"Random Forest Confidence: {result['rf_confidence']:.4f}")

if __name__ == '__main__':
    main()