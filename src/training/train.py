import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
import joblib
from ..utils.dataset import prepare_dataset, get_dataset_statistics
from ...config.config import config

def create_model(input_dim):
    """Create the neural network model"""
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

def train_models(data_dir):
    """Train both neural network and random forest models"""
    # Get dataset statistics
    stats = get_dataset_statistics(data_dir)
    print("\nDataset Statistics:")
    print(f"Total Images: {stats['total_images']}")
    print(f"Pothole Images: {stats['pothole_images']}")
    print(f"Non-Pothole Images: {stats['non_pothole_images']}")
    print(f"Average Image Size: {stats['average_image_size']}")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    X_train, X_test, y_train, y_test, scaler = prepare_dataset(
        data_dir,
        test_size=config['default'].TRAIN_TEST_SPLIT,
        random_state=config['default'].RANDOM_STATE
    )
    
    print(f"\nTraining Set Size: {len(X_train)}")
    print(f"Test Set Size: {len(X_test)}")
    
    # Train Neural Network
    print("\nTraining Neural Network...")
    nn_model = create_model(X_train.shape[1])
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['default'].EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'models/pothole_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    nn_history = nn_model.fit(
        X_train, y_train,
        batch_size=config['default'].BATCH_SIZE,
        epochs=config['default'].EPOCHS,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=config['default'].RANDOM_STATE
    )
    
    rf_model.fit(X_train, y_train)
    
    # Save Random Forest model
    joblib.dump(rf_model, 'models/rf_model.pkl')
    
    # Evaluate models
    print("\nModel Evaluation:")
    
    # Neural Network evaluation
    nn_loss, nn_acc = nn_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nNeural Network:")
    print(f"Test Accuracy: {nn_acc:.4f}")
    print(f"Test Loss: {nn_loss:.4f}")
    
    # Random Forest evaluation
    rf_acc = rf_model.score(X_test, y_test)
    print(f"\nRandom Forest:")
    print(f"Test Accuracy: {rf_acc:.4f}")
    
    # Ensemble evaluation
    nn_pred = nn_model.predict(X_test)
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    ensemble_pred = (0.6 * nn_pred.flatten() + 0.4 * rf_pred) > 0.5
    ensemble_acc = np.mean(ensemble_pred == y_test)
    print(f"\nEnsemble:")
    print(f"Test Accuracy: {ensemble_acc:.4f}")
    
    return nn_model, rf_model, scaler

if __name__ == '__main__':
    # Set the data directory
    data_dir = 'data'
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train models
    nn_model, rf_model, scaler = train_models(data_dir)
    
    print("\nTraining completed successfully!")
    print("Models saved in 'models' directory:")
    print("- Neural Network: pothole_model.h5")
    print("- Random Forest: rf_model.pkl")
    print("- Scaler: scaler.pkl") 