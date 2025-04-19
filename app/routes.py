from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, current_app
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from skimage.feature import hog
from datetime import datetime
from werkzeug.utils import secure_filename
from models import db, UserUpload, ForumPost, Comment
from utils.image_processing import extract_hog_features

# Create blueprint
main_bp = Blueprint('main', __name__)

def load_ml_models():
    """Load machine learning models using config paths"""
    global model, scaler
    model = load_model(current_app.config['MODEL_PATH'])
    scaler = joblib.load(current_app.config['SCALER_PATH'])

def create_forum_post(upload_id, pothole_detected, confidence_score):
    """Create a forum post for the uploaded image"""
    title = f"Pothole Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    content = f"""
    A new pothole has been reported in your area.
    
    Detection Details:
    - Pothole Detected: {'Yes' if pothole_detected else 'No'}
    - Confidence Score: {confidence_score:.2%}
    - Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    
    Please verify this report and take necessary action.
    """
    
    post = ForumPost(
        title=title,
        content=content,
        upload_id=upload_id
    )
    db.session.add(post)
    db.session.commit()
    return post

@main_bp.before_app_first_request
def initialize_models():
    """Load ML models before first request"""
    load_ml_models()

@main_bp.route('/')
def home():
    forum_posts = ForumPost.query.order_by(ForumPost.created_date.desc()).all()
    return render_template('index.html', forum_posts=forum_posts)

@main_bp.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the image
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)

        # Process image
        image = cv2.imread(filepath)
        features = extract_hog_features(image)
        features = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features)[0][0]
        is_pothole = prediction > 0.5
        confidence = float(prediction if is_pothole else 1 - prediction)

        # Save upload record
        upload = UserUpload(
            image_path=saved_filename,
            pothole_detected=is_pothole,
            confidence_score=confidence
        )
        db.session.add(upload)
        db.session.commit()

        # Create forum post
        post = create_forum_post(upload.id, is_pothole, confidence)

        return jsonify({
            'success': True,
            'is_pothole': bool(is_pothole),
            'confidence': confidence,
            'message': 'Pothole detected' if is_pothole else 'No pothole detected',
            'post_id': post.id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/forum/post/<int:post_id>')
def view_post(post_id):
    post = ForumPost.query.get_or_404(post_id)
    return render_template('post.html', post=post)

@main_bp.route('/forum/post/<int:post_id>/comment', methods=['POST'])
def add_comment(post_id):
    post = ForumPost.query.get_or_404(post_id)
    content = request.form.get('content')
    
    if content:
        comment = Comment(content=content, post_id=post_id)
        db.session.add(comment)
        db.session.commit()
        flash('Comment added successfully!', 'success')
    
    return redirect(url_for('main.view_post', post_id=post_id)) 