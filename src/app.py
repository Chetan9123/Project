import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import json

# Import prediction module
from src.prediction.predict import load_model, predict_image

app = Flask(__name__)
app.config.from_object('config.Config')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load model at startup
model = load_model(app.config['MODEL_PATH'])
if model is None:
    print("WARNING: Model could not be loaded. The application may not function correctly.")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and prediction"""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user doesn't select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            file.save(file_path)
            
            # Make prediction
            prediction, result_img = predict_image(model, file_path, threshold=0.5)
            
            if prediction is not None:
                # Save result image
                result_filename = f"result_{saved_filename}"
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                cv2.imwrite(result_path, result_img)
                
                # Determine classification
                is_pothole = prediction >= 0.5
                classification = "Pothole" if is_pothole else "Normal Road"
                
                # Return results
                return render_template(
                    'result.html',
                    original_image=url_for('uploaded_file', filename=saved_filename),
                    result_image=url_for('result_file', filename=result_filename),
                    prediction=prediction,
                    classification=classification
                )
            else:
                flash('Error processing image')
                return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/batch', methods=['GET', 'POST'])
def batch_upload():
    """Handle batch file upload and prediction"""
    if request.method == 'POST':
        # Check if files were uploaded
        if 'files[]' not in request.files:
            flash('No files part')
            return redirect(request.url)
        
        files = request.files.getlist('files[]')
        
        if not files or files[0].filename == '':
            flash('No selected files')
            return redirect(request.url)
        
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                # Save the uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_filename = f"{timestamp}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
                file.save(file_path)
                
                # Make prediction
                prediction, result_img = predict_image(model, file_path, threshold=0.5)
                
                if prediction is not None:
                    # Save result image
                    result_filename = f"result_{saved_filename}"
                    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                    cv2.imwrite(result_path, result_img)
                    
                    # Determine classification
                    is_pothole = prediction >= 0.5
                    classification = "Pothole" if is_pothole else "Normal Road"
                    
                    # Add to results
                    results.append({
                        'filename': filename,
                        'original_image': url_for('uploaded_file', filename=saved_filename),
                        'result_image': url_for('result_file', filename=result_filename),
                        'prediction': float(prediction),
                        'classification': classification
                    })
        
        # Save results to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"batch_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        return render_template('batch_results.html', results=results)
    
    return render_template('batch_upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(file_path)
        
        # Make prediction
        prediction, result_img = predict_image(model, file_path, threshold=0.5)
        
        if prediction is not None:
            # Save result image
            result_filename = f"result_{saved_filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            cv2.imwrite(result_path, result_img)
            
            # Determine classification
            is_pothole = prediction >= 0.5
            classification = "Pothole" if is_pothole else "Normal Road"
            
            # Return results
            return jsonify({
                'filename': filename,
                'prediction': float(prediction),
                'classification': classification,
                'result_image': url_for('result_file', filename=result_filename, _external=True)
            })
    
    return jsonify({'error': 'Error processing image'}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True) 