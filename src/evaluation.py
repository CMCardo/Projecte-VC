import cv2
import numpy as np

def extract_manual_profile(image_path):

    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Convertir a HSV (millor deteccio de color)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Necessitem dos mascaras (180 + 180 graus)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    height, width = red_mask.shape
    profile = np.full(width, np.nan)
    
    # Busquem el pixel vermell mes alt en cada columna
    for x in range(width):
        y_indices = np.where(red_mask[:, x] > 0)[0]
        if len(y_indices) > 0:
            profile[x] = height - y_indices[0] # Measure height from the bottom
            
    return profile


def compare_contours(profile_manual, profile_algo, tolerance_px=5):

    if profile_manual is None or profile_algo is None:
        return 0.0, 0.0, 0.0
        
    # Asegurem que els arrays tenen la mateixa dimensió
    min_len = min(len(profile_manual), len(profile_algo))
    p_man = profile_manual[:min_len]
    p_alg = profile_algo[:min_len]
    
    # Només comparem columnes on el perfil manual te una linea vermella
    valid_cols = ~np.isnan(p_man) & ~np.isnan(p_alg)
    
    p_man_valid = p_man[valid_cols]
    p_alg_valid = p_alg[valid_cols]
    
    if len(p_man_valid) == 0:
        return 0.0, 0.0, 0.0
        
    # Calcula la diferencia dels pixels verticals
    diffs = np.abs(p_man_valid - p_alg_valid)
    
    # Calcul de métriques
    matches = np.sum(diffs <= tolerance_px)
    errors = len(diffs) - matches
    
    match_pct = (matches / len(diffs)) * 100
    error_pct = (errors / len(diffs)) * 100
    mean_dist = np.mean(diffs)
    
    return match_pct, error_pct, mean_dist
