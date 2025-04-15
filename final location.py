from PIL import Image
import exifread

def get_decimal_from_dms(dms, ref):
    degrees = dms[0].num / dms[0].den
    minutes = dms[1].num / dms[1].den
    seconds = dms[2].num / dms[2].den

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_gps(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)

        print("📸 All EXIF Tags:")
        for tag in tags:
            if 'GPS' in tag:
                print(f"{tag}: {tags[tag]}")

        gps_lat = tags.get('GPS GPSLatitude')
        gps_lat_ref = tags.get('GPS GPSLatitudeRef')
        gps_lon = tags.get('GPS GPSLongitude')
        gps_lon_ref = tags.get('GPS GPSLongitudeRef')

        if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
            lat = get_decimal_from_dms(gps_lat.values, gps_lat_ref.values)
            lon = get_decimal_from_dms(gps_lon.values, gps_lon_ref.values)
            return lat, lon
        else:
            print("❌ GPS data not found in EXIF metadata.")
            return None

# Example usage:
image_path = 'C:\\Users\\Gayatri_0949\\OneDrive\\Desktop\\project\\Project\\trialgeo1.jpg.jpg'  # Replace with your image path
coordinates = extract_gps(image_path)
print("Latitude and Longitude:", coordinates)
