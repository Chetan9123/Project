import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the project environment"""
    print("Setting up environment...")
    
    # Create necessary directories
    directories = [
        'data/potholes',
        'data/non_potholes',
        'data/test',
        'models',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Creating .env file...")
        with open('.env', 'w') as f:
            f.write("""SECRET_KEY=your-secret-key-here
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
MAIL_USERNAME=your-email
MAIL_PASSWORD=your-email-password""")
        print("Created .env file. Please update with your actual credentials.")

def check_dataset():
    """Check if dataset is properly set up"""
    print("\nChecking dataset...")
    
    pothole_dir = Path('data/potholes')
    non_pothole_dir = Path('data/non_potholes')
    test_dir = Path('data/test')
    
    pothole_images = list(pothole_dir.glob('*.jpg')) + list(pothole_dir.glob('*.png'))
    non_pothole_images = list(non_pothole_dir.glob('*.jpg')) + list(non_pothole_dir.glob('*.png'))
    test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    
    print(f"Found {len(pothole_images)} pothole images")
    print(f"Found {len(non_pothole_images)} non-pothole images")
    print(f"Found {len(test_images)} test images")
    
    if len(pothole_images) == 0 or len(non_pothole_images) == 0:
        print("Warning: No images found in dataset directories!")
        print("Please add images to:")
        print("- data/potholes/ (for pothole images)")
        print("- data/non_potholes/ (for non-pothole images)")
        print("- data/test/ (for test images)")
        return False
    return True

def run_training():
    """Run the training script"""
    print("\nStarting model training...")
    try:
        subprocess.run([sys.executable, '-m', 'src.training.train'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        return False

def run_testing():
    """Run the testing script"""
    print("\nStarting model testing...")
    try:
        subprocess.run([sys.executable, '-m', 'src.testing.test_model'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during testing: {e}")
        return False

def run_application():
    """Run the main application"""
    print("\nStarting the application...")
    try:
        subprocess.run([sys.executable, 'main.py'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running application: {e}")
        return False

def main():
    """Main function to run the entire project"""
    print("Starting Pothole Detection System...")
    
    # Setup environment
    setup_environment()
    
    # Check dataset
    if not check_dataset():
        print("\nPlease add images to the dataset directories and run again.")
        return
    
    # Run training
    if not run_training():
        print("\nTraining failed. Please check the errors above.")
        return
    
    # Run testing
    if not run_testing():
        print("\nTesting failed. Please check the errors above.")
        return
    
    # Run application
    if not run_application():
        print("\nApplication failed to start. Please check the errors above.")
        return
    
    print("\nProject completed successfully!")

if __name__ == '__main__':
    main() 