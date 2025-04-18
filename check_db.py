import mysql.connector
import cv2
import numpy as np
from tabulate import tabulate  # For tabular display

# Connect to the MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Dell123$",  # Replace with your MySQL password
    database="pothole_db"
)

cursor = conn.cursor()

# Fetch all rows from the detections table
cursor.execute("SELECT id, timestamp, confidence FROM detections")
rows = cursor.fetchall()

# Display results in table format
headers = ["ID", "Timestamp", "Confidence"]
print("\n📊 Detection Records:")
print(tabulate(rows, headers=headers, tablefmt="grid"))

# Fetch images separately and show them
cursor.execute("SELECT id, image_data FROM detections")
image_rows = cursor.fetchall()

for row in image_rows:
    id, image_blob = row
    nparr = np.frombuffer(image_blob, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cv2.imshow(f"Image - ID {id}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cursor.close()
conn.close()
