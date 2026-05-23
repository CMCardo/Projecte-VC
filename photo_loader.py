import cv2
import numpy as np
import os
from rembg import remove, new_session
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def get_image_path(photo_num):

    supported_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".webp", ".WEBP"]
    for ext in supported_extensions:
        ruta = f"Photos/IMG_{photo_num}{ext}"
        if os.path.exists(ruta):
            return ruta
    return None


def sky_remove_cv2(photo_num):

    image_name = get_image_path(photo_num) 
    if not image_name:
        print(f"Error: No s'ha trobat la imatge {photo_num}.")
        return None

    image = cv2.imread(image_name)

    bluescale = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 30, 20]) 
    upper_blue = np.array([140, 255, 255]) 

    sky_mask = cv2.inRange(bluescale, lower_blue, upper_blue)
    mountain_mask = cv2.bitwise_not(sky_mask) 

    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv2.morphologyEx(mountain_mask, cv2.MORPH_OPEN, kernel) #opening
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel) # closing

    result = cv2.bitwise_and(image, image, mask=mask_clean)

    b, g, r = cv2.split(result) 
    rgba = [b, g, r, mask_clean]
    transparet_result = cv2.merge(rgba)

    if not os.path.exists("sky_remove_cv2"):
        os.makedirs("sky_remove_cv2")

    cv2.imwrite(f'sky_remove_cv2/mountain_{photo_num}_color.png', transparet_result)


def sky_remove_ai(photo_num, model):

    #image route
    image_name = get_image_path(photo_num) 
    output_name = f"sky_remove_ai/mountain_{photo_num}_cutted.png"

    if not os.path.exists("sky_remove_ai"):
        os.makedirs("sky_remove_ai")

    # comprovation if extension is in lowercase
    if not image_name:
        print(f"Error: Image not found '{image_name}'.")
        return

    print("\nProcessing image with AI")
    print(f"Model: {model}")
    print("Firts time it will take a while, it'll download the model")
    
    input_image = Image.open(image_name)
    
    session = new_session(model)
    
    output_image = remove(input_image, session=session)
    output_image.save(output_name)
    
    print(f"\n Image saved '{output_name}'")
    

#Laplace

def sky_remove_Laplace(photo_num, tolerance):

    image_name = get_image_path(photo_num) 
    
    if not image_name:
        print(f"Error: No s'ha trobat la imatge '{photo_num}'.")
        return None
            
    image = cv2.imread(image_name)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (9, 9)) #apply blur to delete image noise
    
    #the sky have smooth zones with values ​​near to 0, the other zones with texture will have high values
    laplacian = cv2.Laplacian(blur, cv2.CV_8U)
    
    _, cel_mask = cv2.threshold(laplacian, tolerance, 255, cv2.THRESH_BINARY_INV) #if the texture value is < , tolerance convert to white
    
    kernel = np.ones((15,15), np.uint8)
    mask_clean = cv2.morphologyEx(cel_mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    
    #invert colors
    mountain_mask = cv2.bitwise_not(mask_clean)
    
    #add to rgb alfa channel for transparency
    result = cv2.bitwise_and(image, image, mask=mountain_mask)
    b, g, r = cv2.split(result) 
    rgba = [b, g, r, mountain_mask]
    transparet_result = cv2.merge(rgba)

    if not os.path.exists("Laplace"):
        os.makedirs("Laplace")

    output_rute = f'Laplace/mountain_{photo_num}_texture.png'
    cv2.imwrite(output_rute, transparet_result)
    
    return output_rute



# Laplace + cv2

def sky_remove_hybrid(photo_num, tolerance):
    image_name = get_image_path(photo_num) 
    if not image_name:
        return None
            
    image = cv2.imread(image_name)
    
    # hsv mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 30, 20]) 
    upper_blue = np.array([140, 255, 255]) 
    mask_color = cv2.inRange(hsv, lower_blue, upper_blue) # Blue = white
    
    # texture mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (9, 9))
    laplacian = cv2.Laplacian(blur, cv2.CV_8U)
    _, mask_texture = cv2.threshold(laplacian, tolerance, 255, cv2.THRESH_BINARY_INV) # smooth = white
    
    # intersection
    combined_sky_mask = cv2.bitwise_and(mask_color, mask_texture) # if the 2 mask has white assign white at the final pixel
    
    kernel = np.ones((5,5), np.uint8)
    combined_sky_mask = cv2.morphologyEx(combined_sky_mask, cv2.MORPH_OPEN, kernel)
    combined_sky_mask = cv2.morphologyEx(combined_sky_mask, cv2.MORPH_CLOSE, kernel)
    
    # invert the image to have the mountain
    mountain_mask = cv2.bitwise_not(combined_sky_mask)
    
    result = cv2.bitwise_and(image, image, mask=mountain_mask)
    b, g, r = cv2.split(result) 
    rgba = [b, g, r, mountain_mask]
    transparet_result = cv2.merge(rgba)

    if not os.path.exists("Hybrid"):
        os.makedirs("Hybrid")

    output_rute = f'Hybrid/mountain_{photo_num}_hybrid.png'
    cv2.imwrite(output_rute, transparet_result)
    return output_rute




def get_gps_from_photo(image_path):

    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if not exif_data:
            return None
        
        # search for gps metadata
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
        print(f"Error llegint el GPS de la foto: {e}")
        return None