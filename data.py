import os
import csv

# Define folders with their corresponding labels
folders = {
    "normal": r"C:/Users/Admin/Desktop/hiproject/normal",
    "pothole": r"C:/Users/Admin/Desktop/hiproject/potholes"
}

csv_file = "dataa.csv"

# Open CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Label", "image_path"])  # CSV headers

    for label, folder_path in folders.items():
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found!")
            continue  # Skip this folder if it doesn't exist

        # Get image files from the folder
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]

        for image in image_files:
            image_path = os.path.join(folder_path, image)
            writer.writerow([label, image_path])  # Write label and path

        print(f"Added {len(image_files)} images from '{folder_path}' under label '{label}'.")

print(f"CSV file '{csv_file}' created successfully!")
