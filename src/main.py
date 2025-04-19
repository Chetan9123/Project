import os
import sys
from flask import Flask
from config import config
from src.models import db
from src.utils.dataset import prepare_dataset
from src.training.train import train_models
from src.testing.test_model import evaluate_models, test_single_image

def create_app(config_name='default'):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app

def main():
    """Main entry point for the application"""
    # Create the Flask application
    app = create_app()
    
    # Set up the data directory structure
    os.makedirs('data/potholes', exist_ok=True)
    os.makedirs('data/non_potholes', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Check if models exist, if not train them
    if not (os.path.exists('models/pothole_model.h5') and 
            os.path.exists('models/rf_model.pkl') and 
            os.path.exists('models/scaler.pkl')):
        print("Training models...")
        train_models('data')
    
    # Run the Flask application
    app.run(debug=True)

if __name__ == '__main__':
    main() 