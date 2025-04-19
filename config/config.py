import os

class Config:
    # Base directory of the project
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Model paths
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pothole_model.h5')
    SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    RF_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Database settings
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'pothole.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Google OAuth settings
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
    GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
    
    # API settings
    API_VERSION = '1.0'
    API_TITLE = 'Pothole Detection API'
    API_DESCRIPTION = 'API for detecting potholes in road images'
    
    # Model training settings
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_STATE = 42
    BATCH_SIZE = 32
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
    # Model Architecture
    NEURAL_NETWORK_LAYERS = [512, 256, 128, 64]
    DROPOUT_RATES = [0.3, 0.3, 0.2, 0.2]
    LEARNING_RATE = 0.001
    
    # Random Forest Parameters
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    RF_MIN_SAMPLES_SPLIT = 5
    RF_MIN_SAMPLES_LEAF = 2
    
    # Ensemble Weights
    NN_WEIGHT = 0.6
    RF_WEIGHT = 0.4
    
    # Email settings
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', True)
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')
    
    # Ensure required directories exist
    @staticmethod
    def init_app(app):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(os.path.dirname(app.config['MODEL_PATH']), exist_ok=True)
        
class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    # Add production-specific settings here
    
class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 