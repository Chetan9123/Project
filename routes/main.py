from flask import Blueprint, render_template, request, jsonify, current_app, flash, redirect, url_for, send_from_directory
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from datetime import datetime
from src.models.models import db, PotholeDetection, User
from src.utils.image_processing import process_image, extract_features
from src.utils.model import load_model, predict_pothole
from forms.auth import LoginForm, RegistrationForm, ResetPasswordRequestForm, ResetPasswordForm

# Create blueprint
main_bp = Blueprint('main', __name__)

# Load models at startup
nn_model, rf_model, scaler = None, None, None

@main_bp.before_app_first_request
def initialize_models():
    """Load ML models before first request"""
    global nn_model, rf_model, scaler
    nn_model, rf_model, scaler = load_models()

@main_bp.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@main_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Single image upload route"""
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], saved_filename)
            file.save(filepath)
            
            # Process image and make prediction
            processed_image = process_image(filepath)
            features = extract_features(processed_image)
            model = load_model()
            prediction, confidence = predict_pothole(model, features)
            
            # Save detection record
            detection = PotholeDetection(
                image_path=saved_filename,
                pothole_detected=prediction,
                confidence_score=confidence,
                user_id=current_user.id
            )
            db.session.add(detection)
            db.session.commit()
            
            return render_template('result.html', 
                                detection=detection,
                                image_path=filepath)
    
    return render_template('upload.html')

@main_bp.route('/batch_upload', methods=['GET', 'POST'])
@login_required
def batch_upload():
    """Batch image upload route"""
    if request.method == 'POST':
        if 'images[]' not in request.files:
            return jsonify({'error': 'No images uploaded'}), 400
            
        files = request.files.getlist('images[]')
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                saved_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], saved_filename)
                file.save(filepath)
                
                # Process image and make prediction
                processed_image = process_image(filepath)
                features = extract_features(processed_image)
                model = load_model()
                prediction, confidence = predict_pothole(model, features)
                
                # Save detection record
                detection = PotholeDetection(
                    image_path=saved_filename,
                    pothole_detected=prediction,
                    confidence_score=confidence,
                    user_id=current_user.id
                )
                db.session.add(detection)
                results.append({
                    'filename': filename,
                    'prediction': prediction,
                    'confidence': confidence
                })
        
        db.session.commit()
        return jsonify({'results': results})
        
    return render_template('batch_upload.html')

@main_bp.route('/history')
@login_required
def history():
    """User detection history route"""
    detections = PotholeDetection.query.filter_by(user_id=current_user.id)\
        .order_by(PotholeDetection.created_at.desc()).all()
    return render_template('history.html', detections=detections)

@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

@main_bp.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory(current_app.config['RESULTS_FOLDER'], filename)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS'] 