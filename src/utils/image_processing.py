import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import albumentations as A
from PIL import Image, ImageDraw, ImageFont
import os

def process_image(image_path):
    """Process image for feature extraction"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return thresh

def extract_features(img):
    """Extract features from processed image"""
    # Calculate basic statistics
    mean = np.mean(img)
    std = np.std(img)
    
    # Calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Calculate texture features using GLCM
    glcm = cv2.cornerHarris(img, 2, 3, 0.04)
    harris_response = np.mean(glcm)
    
    # Calculate edge features
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.mean(edges > 0)
    
    # Combine features
    features = np.concatenate([
        [mean, std, harris_response, edge_density],
        hist
    ])
    
    return features

def draw_detection(image_path, is_pothole, confidence):
    """Draw detection results on image"""
    # Open original image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Set colors and text
    color = (255, 0, 0) if is_pothole else (0, 255, 0)
    text = f"{'Pothole' if is_pothole else 'No Pothole'} ({confidence:.2%})"
    
    # Draw border
    width, height = img.size
    border_width = 5
    draw.rectangle(
        [(0, 0), (width-1, height-1)],
        outline=color,
        width=border_width
    )
    
    # Draw text background
    text_bbox = draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding = 10
    
    draw.rectangle(
        [(0, 0), (text_width + 2*padding, text_height + 2*padding)],
        fill=(0, 0, 0, 128)
    )
    
    # Draw text
    draw.text(
        (padding, padding),
        text,
        fill=color
    )
    
    return img

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