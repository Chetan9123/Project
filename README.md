# Pothole Detection System

A machine learning-based system for detecting potholes in road images using both Neural Network and Random Forest models.

## Project Structure

```
Project/
├── config/              # Configuration files
├── data/               # Dataset directory
│   ├── potholes/      # Pothole images
│   ├── non_potholes/  # Non-pothole images
│   └── test/          # Test images
├── logs/              # Log files
├── models/            # Trained models
├── src/               # Source code
│   ├── models/       # Database models
│   ├── training/     # Training scripts
│   ├── testing/      # Testing scripts
│   └── utils/        # Utility functions
└── main.py           # Application entry point
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```
SECRET_KEY=your-secret-key
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
MAIL_USERNAME=your-email
MAIL_PASSWORD=your-email-password
```

4. Prepare the dataset:
- Place pothole images in `data/potholes/`
- Place non-pothole images in `data/non_potholes/`
- Place test images in `data/test/`

5. Run the application:
```bash
python main.py
```

## Features

- Dual model approach (Neural Network + Random Forest)
- Image preprocessing and feature extraction
- Model training and evaluation
- Web interface for image upload and prediction
- User authentication (Email/Password + Google OAuth)
- Detailed performance metrics and visualizations

## Testing

To test the models:
```bash
python src/testing/test_model.py
```

This will:
- Load trained models
- Evaluate on test dataset
- Generate performance metrics
- Test individual images
- Create visualizations

## Training

To train the models:
```bash
python src/training/train.py
```

This will:
- Prepare the dataset
- Train both models
- Save models and scaler
- Generate training metrics

## License

MIT License
