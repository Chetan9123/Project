import os
import numpy as np
from pathlib import Path
import tensorflow as tf
import cv2
import argparse
from datetime import datetime

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            return None, None
        
        # Store original image for visualization
        original_img = img.copy()
        
        # Preprocess for model
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0
        
        return img, original_img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None

def load_model(model_path):
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_image(model, image_path, threshold=0.5):
    """Make prediction on a single image"""
    # Load and preprocess image
    img, original_img = load_and_preprocess_image(image_path)
    if img is None:
        return None, None
    
    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0))[0][0]
    
    # Determine class
    is_pothole = prediction >= threshold
    
    # Create visualization
    result_img = original_img.copy()
    text = f"Pothole: {prediction:.2f}" if is_pothole else f"Normal: {1-prediction:.2f}"
    color = (0, 0, 255) if is_pothole else (0, 255, 0)  # Red for pothole, Green for normal
    
    # Add text to image
    cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return prediction, result_img

def process_directory(model, input_dir, output_dir, threshold=0.5):
    """Process all images in a directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"predictions_{timestamp}.csv"
    
    with open(results_file, "w") as f:
        f.write("Image,Prediction,Is_Pothole\n")
        
        # Process each image
        image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        print(f"Processing {len(image_files)} images...")
        
        for img_path in image_files:
            prediction, result_img = predict_image(model, img_path, threshold)
            
            if prediction is not None:
                # Save result image
                output_path = output_dir / f"result_{img_path.name}"
                cv2.imwrite(str(output_path), result_img)
                
                # Write to results file
                is_pothole = prediction >= threshold
                f.write(f"{img_path.name},{prediction:.4f},{is_pothole}\n")
                
                print(f"Processed {img_path.name}: {'Pothole' if is_pothole else 'Normal'} ({prediction:.4f})")
    
    print(f"Results saved to {results_file}")
    return results_file

def main():
    parser = argparse.ArgumentParser(description="Pothole Detection System")
    parser.add_argument("--model", type=str, default="models/pothole_model.h5", help="Path to the trained model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, default="results", help="Path to output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return
    
    # Check if input is a directory or single file
    input_path = Path(args.input)
    if input_path.is_dir():
        # Process directory
        results_file = process_directory(model, input_path, args.output, args.threshold)
        print(f"Processed all images in {input_path}")
        print(f"Results saved to {results_file}")
    else:
        # Process single image
        prediction, result_img = predict_image(model, input_path, args.threshold)
        
        if prediction is not None:
            # Create output directory if it doesn't exist
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            
            # Save result image
            output_path = output_dir / f"result_{input_path.name}"
            cv2.imwrite(str(output_path), result_img)
            
            is_pothole = prediction >= args.threshold
            print(f"Image: {input_path.name}")
            print(f"Prediction: {prediction:.4f}")
            print(f"Classification: {'Pothole' if is_pothole else 'Normal'}")
            print(f"Result image saved to {output_path}")

if __name__ == "__main__":
    main() 