import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import albumentations as A

def extract_features(image):
    """Extract multiple feature sets from an image"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Resize image to standard size
    resized = cv2.resize(gray, (128, 128))
    
    # Extract HOG features with improved parameters
    hog_features = hog(resized, 
                      orientations=12,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      visualize=False)
    
    # Extract LBP features
    lbp = local_binary_pattern(resized, 8, 1, method='uniform')
    lbp_hist = np.histogram(lbp, bins=32, range=(0, 32))[0]
    
    # Extract texture features using GLCM
    glcm = cv2.calcHist([resized], [0], None, [256], [0, 256])
    glcm = glcm.flatten() / glcm.sum()
    
    # Combine all features
    features = np.concatenate([hog_features, lbp_hist, glcm])
    return features

def preprocess_image(image_path):
    """Load and preprocess image for prediction with augmentation"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    
    # Define augmentation pipeline
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(height=128, width=128, p=0.5),
    ])
    
    # Apply augmentation
    augmented = transform(image=image)['image']
    
    # Extract features from both original and augmented images
    original_features = extract_features(image)
    augmented_features = extract_features(augmented)
    
    # Average the features
    final_features = (original_features + augmented_features) / 2
    
    return final_features

def enhance_image(image):
    """Apply image enhancement techniques"""
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    # Apply Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
    
    # Normalize the image
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    return enhanced 