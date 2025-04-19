from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from src.utils.image_processing import process_image, extract_features, draw_detection
from src.utils.model import load_model, predict_pothole
from config.config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and scaler
model, scaler = load_model(app.config['MODEL_PATH'], app.config['SCALER_PATH'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image and make prediction
        processed_img = process_image(filepath)
        features = extract_features(processed_img)
        is_pothole, confidence = predict_pothole(model, scaler, features)
        
        # Draw detection on image
        result_img = draw_detection(filepath, is_pothole, confidence)
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
        result_img.save(result_path)
        
        return jsonify({
            'success': True,
            'is_pothole': is_pothole,
            'confidence': confidence,
            'result_image': f'result_{filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/batch_upload', methods=['GET', 'POST'])
def batch_upload():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files part'}), 400
            
        files = request.files.getlist('files[]')
        results = []
        
        for file in files:
            if file.filename == '':
                continue
                
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process image and make prediction
                processed_img = process_image(filepath)
                features = extract_features(processed_img)
                is_pothole, confidence = predict_pothole(model, scaler, features)
                
                # Draw detection on image
                result_img = draw_detection(filepath, is_pothole, confidence)
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
                result_img.save(result_path)
                
                results.append({
                    'filename': filename,
                    'is_pothole': is_pothole,
                    'confidence': confidence,
                    'result_image': f'result_{filename}'
                })
        
        return jsonify({'success': True, 'results': results})
        
    return render_template('batch_upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True) 