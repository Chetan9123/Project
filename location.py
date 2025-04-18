# Import necessary libraries
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pandas as pd

# Function to extract GPS data
def get_geotagging(exif):
    if not exif:
        return None
    
    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            gps_info = exif.get(idx)
            if gps_info:
                for key, val in gps_info.items():
                    geotagging[GPSTAGS.get(key, key)] = val
    return geotagging

# Function to get latitude and longitude
def get_coordinates(geotags):
    def to_degrees(value):
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)

    if 'GPSLatitude' in geotags and 'GPSLatitudeRef' in geotags:
        lat = to_degrees(geotags['GPSLatitude'])
        if geotags['GPSLatitudeRef'] != 'N':
            lat = -lat
    else:
        lat = None

    if 'GPSLongitude' in geotags and 'GPSLongitudeRef' in geotags:
        lon = to_degrees(geotags['GPSLongitude'])
        if geotags['GPSLongitudeRef'] != 'E':
            lon = -lon
    else:
        lon = None
    
    return lat, lon

# Main block to process the image
image_path = "C:\\Users\\Admin\\Documents\\trial\\trialpic1.jpg"  # Replace with your image path
try:
    # Load the image and extract EXIF data
    image = Image.open(image_path)
    exif_data = image._getexif()
    
    # Extract geotags
    if exif_data:
        geotags = get_geotagging(exif_data)
        if geotags:
            latitude, longitude = get_coordinates(geotags)
            print("Latitude:", latitude)
            print("Longitude:", longitude)
        else:
            print("No geotagging found in the image.")
    else:
        print("No EXIF data found in the image.")
except FileNotFoundError:
    print("The specified image file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Save the coordinates in a CSV file (if available)
if 'latitude' in locals() and 'longitude' in locals():
    data = {"Image Name": [image_path], "Latitude": [latitude], "Longitude": [longitude]}
    df = pd.DataFrame(data)
    df.to_csv("geotagged_data.csv", index=False)
    print("Data saved to geotagged_data.csv")
