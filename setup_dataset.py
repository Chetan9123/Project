import os
import shutil
from pathlib import Path
import random

def setup_dataset():
    """Move images from potholes and normal directories to the project data directories"""
    print("Setting up dataset...")
    
    # Source directories
    pothole_src = "potholes"
    normal_src = "normal"
    
    # Destination directories
    pothole_dest = "data/potholes"
    normal_dest = "data/non_potholes"
    test_dest = "data/test"
    
    # Create destination directories if they don't exist
    for dir_path in [pothole_dest, normal_dest, test_dest]:
        os.makedirs(dir_path, exist_ok=True)
    
    def copy_files(src_dir, dest_dir, test_ratio=0.2):
        if not os.path.exists(src_dir):
            print(f"\nSource directory {src_dir} does not exist!")
            return 0, 0
        
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)
        
        split_idx = int(len(files) * (1 - test_ratio))
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        # Copy training files
        for file in train_files:
            src_path = os.path.join(src_dir, file)
            dest_path = os.path.join(dest_dir, file)
            shutil.copy2(src_path, dest_path)
            print(f"Copied {file} to training set")
        
        # Copy test files
        for file in test_files:
            src_path = os.path.join(src_dir, file)
            dest_path = os.path.join(test_dest, file)
            shutil.copy2(src_path, dest_path)
            print(f"Copied {file} to test set")
        
        return len(train_files), len(test_files)
    
    print("\nCopying pothole images...")
    pothole_train, pothole_test = copy_files(pothole_src, pothole_dest)
    
    print("\nCopying normal (non-pothole) images...")
    normal_train, normal_test = copy_files(normal_src, normal_dest)
    
    print("\nDataset setup completed!")
    print(f"Training pothole images: {pothole_train}")
    print(f"Training normal images: {normal_train}")
    print(f"Test images: {pothole_test + normal_test}")

if __name__ == "__main__":
    setup_dataset() 