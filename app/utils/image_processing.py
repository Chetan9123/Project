import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def process_image(image_path):
    """
    Process an image for pothole detection.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Processed image array
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to standard size
    img = cv2.resize(img, (224, 224))
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    return img

def extract_features(img):
    """
    Extract features from a processed image.
    
    Args:
        img (numpy.ndarray): Processed image array
        
    Returns:
        numpy.ndarray: Feature vector
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        if img.dtype == np.float32:
            # If image is already normalized
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.float32) / 255.0
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Calculate basic statistics
    mean = np.mean(gray)
    std = np.std(gray)
    
    # Calculate texture features using GLCM
    glcm = calculate_glcm(gray)
    contrast = calculate_contrast(glcm)
    correlation = calculate_correlation(glcm)
    energy = calculate_energy(glcm)
    homogeneity = calculate_homogeneity(glcm)
    
    # Add more features
    edges = cv2.Canny((gray * 255).astype(np.uint8), 100, 200)
    edge_density = np.mean(edges > 0)
    
    # Calculate histogram features
    hist = cv2.calcHist([(gray * 255).astype(np.uint8)], [0], None, [8], [0, 256])
    hist_features = hist.flatten() / np.sum(hist)  # Normalize histogram
    
    # Combine all features
    features = np.concatenate([
        [mean, std, contrast, correlation, energy, homogeneity, edge_density],
        hist_features
    ])
    
    return features

def calculate_glcm(img, levels=8):
    """Calculate Gray-Level Co-occurrence Matrix."""
    # Quantize image to 8 levels
    img_quantized = (img * (levels - 1)).astype(np.uint8)
    
    # Initialize GLCM
    glcm = np.zeros((levels, levels))
    
    # Calculate GLCM
    for i in range(img_quantized.shape[0] - 1):
        for j in range(img_quantized.shape[1] - 1):
            glcm[img_quantized[i, j], img_quantized[i + 1, j + 1]] += 1
    
    # Normalize GLCM
    glcm = glcm / glcm.sum()
    
    return glcm

def calculate_contrast(glcm):
    """Calculate contrast from GLCM."""
    rows, cols = glcm.shape
    contrast = 0
    for i in range(rows):
        for j in range(cols):
            contrast += glcm[i, j] * (i - j) ** 2
    return contrast

def calculate_correlation(glcm):
    """Calculate correlation from GLCM."""
    rows, cols = glcm.shape
    mean_i = np.sum(np.arange(rows) * np.sum(glcm, axis=1))
    mean_j = np.sum(np.arange(cols) * np.sum(glcm, axis=0))
    std_i = np.sqrt(np.sum((np.arange(rows) - mean_i) ** 2 * np.sum(glcm, axis=1)))
    std_j = np.sqrt(np.sum((np.arange(cols) - mean_j) ** 2 * np.sum(glcm, axis=0)))
    
    correlation = 0
    for i in range(rows):
        for j in range(cols):
            correlation += glcm[i, j] * (i - mean_i) * (j - mean_j)
    
    if std_i * std_j == 0:
        return 0
    return correlation / (std_i * std_j)

def calculate_energy(glcm):
    """Calculate energy from GLCM."""
    return np.sum(glcm ** 2)

def calculate_homogeneity(glcm):
    """Calculate homogeneity from GLCM."""
    rows, cols = glcm.shape
    homogeneity = 0
    for i in range(rows):
        for j in range(cols):
            homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
    return homogeneity

def draw_detection(image_path, is_pothole, confidence):
    """
    Draw detection results on the image.
    
    Args:
        image_path (str): Path to the original image
        is_pothole (bool): Whether a pothole was detected
        confidence (float): Confidence score of the detection
        
    Returns:
        PIL.Image: Image with detection visualization
    """
    # Read image
    img = Image.open(image_path)
    
    # Convert to numpy array for OpenCV
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Add text
    text = f"{'Pothole' if is_pothole else 'No Pothole'} ({confidence:.2%})"
    color = (0, 255, 0) if is_pothole else (0, 0, 255)
    
    cv2.putText(img_cv, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Convert back to PIL Image
    result_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    return result_img 