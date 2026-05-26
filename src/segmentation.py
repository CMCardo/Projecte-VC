import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

def sky_remove_cv2(image_path):
    """
    Elimina els cels blaus fen servir el threshold de color amb HSV de OpenCV, reorna
    un array de numpy amb el cel transparent, o None si hi ha error.
    """
    image = cv2.imread(image_path)
    if image is None: return None

    # Conversió al espai de color HSV per millor segmentació de color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 30, 20]) 
    upper_blue = np.array([140, 255, 255]) 

    sky_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mountain_mask = cv2.bitwise_not(sky_mask) 

    # Eliminar soroll de la mascara
    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv2.morphologyEx(mountain_mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel) 

    result = cv2.bitwise_and(image, image, mask=mask_clean)

    # Ajuntem els canals afegin alpha per la transparencia
    b, g, r = cv2.split(result) 
    bgra = cv2.merge([b, g, r, mask_clean])

    return bgra


def sky_remove_ai(image_path, model_name="u2net"):
    """
    Elimina el fons fen servir un model de deep learning ja entrenat (rembg), retorna
    un array de NumPy (BGRA) amb el cel transparent o None si hi ha error.
    """
    try:
        input_image = Image.open(image_path)
        session = new_session(model_name)
        output_image_pil = remove(input_image, session=session)
        
        # Conversió de imatge PIL (RGBA) a OpenCV (BGRA)
        open_cv_image = np.array(output_image_pil) 
        bgra_image = open_cv_image[:, :, [2, 1, 0, 3]] 
        
        return bgra_image
    except Exception as e:
        print(f"Error in AI segmentation: {e}")
        return None


def sky_remove_laplace(image_path, tolerance=3):
    """
    Elimina les areas suaus (com els cels) fen servir detecció de contorns amb Laplace.
    Retorna un array de numpy (BGRA) amb cels transparents o None si hi ha error.
    """
    image = cv2.imread(image_path)
    if image is None: return None
            
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (9, 9)) 
    
    laplacian = cv2.Laplacian(blur, cv2.CV_8U)
    
    # Les areas suaus tindran menys que el threshold (per exemple els cels).
    _, sky_mask = cv2.threshold(laplacian, tolerance, 255, cv2.THRESH_BINARY_INV) 
    
    kernel = np.ones((15,15), np.uint8)
    mask_clean = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    
    mountain_mask = cv2.bitwise_not(mask_clean)
    
    result = cv2.bitwise_and(image, image, mask=mountain_mask)
    b, g, r = cv2.split(result) 
    bgra = cv2.merge([b, g, r, mountain_mask])

    return bgra


def sky_remove_hybrid(image_path, tolerance=3):
    """
    Combina color i textura per una extracció del cel més robusta, retorna un array
    de Numpy (BGRA) amb cel transparent o None si hi ha error.
    """
    image = cv2.imread(image_path)
    if image is None: return None
            
    # Mascara de colord
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 30, 20]) 
    upper_blue = np.array([140, 255, 255]) 
    sky_mask_color = cv2.inRange(hsv, lower_blue, upper_blue) 
    
    # Mascara de textura
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (9, 9))
    laplacian = cv2.Laplacian(blur, cv2.CV_8U)
    _, sky_mask_texture = cv2.threshold(laplacian, tolerance, 255, cv2.THRESH_BINARY_INV) 
    
    # Intersecció, si el pixel es blau i es suau, es decideix com cel
    combined_sky_mask = cv2.bitwise_and(sky_mask_color, sky_mask_texture) 
    
    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv2.morphologyEx(combined_sky_mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    
    mountain_mask = cv2.bitwise_not(mask_clean)
    
    result = cv2.bitwise_and(image, image, mask=mountain_mask)
    b, g, r = cv2.split(result) 
    bgra = cv2.merge([b, g, r, mountain_mask])

    return bgra