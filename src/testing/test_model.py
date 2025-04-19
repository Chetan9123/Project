import os
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from utils.dataset import prepare_dataset, load_and_prepare_single_image
from config import config

def load_models():
    """Load the trained neural network and random forest models"""
    try:
        nn_model = load_model('models/pothole_model.h5')
        rf_model = joblib.load('models/rf_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
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
    plt.savefig('models/confusion_matrices.png')
    plt.close()
    
    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    
    # Neural Network ROC
    fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_pred_proba)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {roc_auc_nn:.2f})')
    
    # Random Forest ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred_proba)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    
    # Ensemble ROC
    fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble_pred_proba)
    roc_auc_ens = auc(fpr_ens, tpr_ens)
    plt.plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC = {roc_auc_ens:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('models/roc_curves.png')
    plt.close()

def test_single_image(image_path, nn_model, rf_model, scaler):
    """Test a single image using both models"""
    try:
        # Prepare the image
        features = load_and_prepare_single_image(image_path)
        
        # Get predictions
        nn_pred_proba = nn_model.predict(features)[0][0]
        rf_pred_proba = rf_model.predict_proba(features)[0][1]
        
        # Ensemble prediction
        ensemble_pred_proba = (config['default'].NN_WEIGHT * nn_pred_proba + 
                             config['default'].RF_WEIGHT * rf_pred_proba)
        
        # Print results
        print(f"\nResults for image: {image_path}")
        print(f"Neural Network probability: {nn_pred_proba:.4f}")
        print(f"Random Forest probability: {rf_pred_proba:.4f}")
        print(f"Ensemble probability: {ensemble_pred_proba:.4f}")
        print(f"Final prediction: {'Pothole' if ensemble_pred_proba > 0.5 else 'No Pothole'}")
        
        return ensemble_pred_proba > 0.5
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def main():
    # Load models
    print("Loading models...")
    nn_model, rf_model, scaler = load_models()
    if nn_model is None or rf_model is None or scaler is None:
        return
    
    # Prepare test dataset
    print("\nPreparing test dataset...")
    X_train, X_test, y_train, y_test, _ = prepare_dataset(
        'data',
        test_size=config['default'].TRAIN_TEST_SPLIT,
        random_state=config['default'].RANDOM_STATE
    )
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluate_models(X_test, y_test, nn_model, rf_model)
    
    # Test single images if provided
    test_images = [
        'data/test/pothole1.jpg',
        'data/test/non_pothole1.jpg'
    ]
    
    print("\nTesting single images...")
    for image_path in test_images:
        if os.path.exists(image_path):
            test_single_image(image_path, nn_model, rf_model, scaler)
        else:
            print(f"Image not found: {image_path}")

if __name__ == '__main__':
    main()