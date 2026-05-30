import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_gps_from_photo(image_path):

    
    try:
        if not os.path.exists(image_path):
            return None

        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if not exif_data:
            return None
        
        gps_info = {}
        for tag, value in exif_data.items():
            decoded_tag = TAGS.get(tag, tag)
            if decoded_tag == "GPSInfo":
                for t in value:
                    gps_tag = GPSTAGS.get(t, t)
                    gps_info[gps_tag] = value[t]
                    
        if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
            return None
            
        def to_decimal(value):
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
            
        lat = to_decimal(gps_info["GPSLatitude"])
        if gps_info.get("GPSLatitudeRef") != "N":
            lat = -lat
            
        lon = to_decimal(gps_info["GPSLongitude"])
        if gps_info.get("GPSLongitudeRef") != "E":
            lon = -lon
            
        alt = 0.0
        if "GPSAltitude" in gps_info:
            alt = float(gps_info["GPSAltitude"])
            
        return lat, lon, alt
        
    except Exception as e:
        print(f"Error reading GPS from photo {image_path}: {e}")
        return None
