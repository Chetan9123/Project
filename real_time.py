import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import time
import mysql.connector
import io
import exifread

# === Load model and scaler ===
model = load_model("C:/Users/Admin/Desktop/hiproject/Project/pothole_detector.h5")
scaler = joblib.load("C:/Users/Admin/Desktop/hiproject/Project/scaler.pkl")

# === GPS Helper Functions ===
def get_decimal_from_dms(dms, ref):
    degrees = dms[0].num / dms[0].den
    minutes = dms[1].num / dms[1].den
    seconds = dms[2].num / dms[2].den
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_gps_from_bytes(image_bytes):
    try:
        f = io.BytesIO(image_bytes)
        tags = exifread.process_file(f, stop_tag="GPS GPSLongitude")
        gps_lat = tags.get('GPS GPSLatitude')
        gps_lat_ref = tags.get('GPS GPSLatitudeRef')
        gps_lon = tags.get('GPS GPSLongitude')
        gps_lon_ref = tags.get('GPS GPSLongitudeRef')
        if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
            lat = get_decimal_from_dms(gps_lat.values, gps_lat_ref.values)
            lon = get_decimal_from_dms(gps_lon.values, gps_lon_ref.values)
            return lat, lon
    except:
        pass
    return None, None

# === Feature Extraction ===
def extract_hog_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return features

def detect_pothole_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pothole_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 5000:
            pothole_contours.append(contour)
    return pothole_contours

# === Connect to MySQL ===
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Dell123$",
    database="pothole_db"
)
cursor = conn.cursor()

# === Open video stream ===
url = "https://100.75.50.72:8080"
cap = cv2.VideoCapture(url)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from camera.")
        break

    try:
        features = extract_hog_features(frame)
        features = scaler.transform([features])
        pred = model.predict(features)[0][0]

        label = "POTHOLE" if pred > 0.5 else "NORMAL"
        color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)

        if label == "POTHOLE":
            contours = detect_pothole_contours(frame)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(frame, "POTHOLE DETECTED!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Convert frame to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()

            # Extract GPS (if available)
            lat, lon = extract_gps_from_bytes(image_data)

            # Insert to DB
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            sql = """
                INSERT INTO detections (timestamp, image_data, confidence, latitude, longitude)
                VALUES (%s, %s, %s, %s, %s)
            """
            values = (timestamp, image_data, float(pred), lat, lon)
            cursor.execute(sql, values)
            conn.commit()
            print(f"✅ Stored frame to DB at {timestamp} | Lat: {lat}, Lon: {lon}")

        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Real-time Pothole Detection", frame)

    except Exception as e:
        print("⚠️ Error:", e)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cursor.close()
conn.close()
cap.release()
cv2.destroyAllWindows()
