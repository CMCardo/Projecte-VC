import cv2
import numpy as np
import pandas as pd

def get_1d_profile(bgra_image):
    if bgra_image is None or bgra_image.shape[2] != 4:
        return None
    
    height, width = bgra_image.shape[:2]
    profile_1d = np.zeros(width)
    alpha_channel = bgra_image[:, :, 3].copy()
    
    # Filtre anti soroll (elimina nubols)
    contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(alpha_channel)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        alpha_channel = clean_mask
    
    # Extracció del pixel mes alt
    for x in range(width):
        y_indices = np.where(alpha_channel[:, x] > 10)[0]
        if len(y_indices) == 0:
            profile_1d[x] = 0 
        else:
            y_mountain = y_indices[0]
            profile_1d[x] = height - y_mountain
            
    # Filtre de mediana
    profile_1d = pd.Series(profile_1d).rolling(window=11, center=True, min_periods=1).median().values
    
    # Suavitzat amb gausiana
    profile_1d = cv2.GaussianBlur(profile_1d.astype(np.float32).reshape(1, -1), (15, 1), 0)[0]
    
    return profile_1d

def draw_mountain_contour(bgra_image):
    if bgra_image is None or bgra_image.shape[2] != 4:
        return None
        
    alpha_channel = bgra_image[:, :, 3].copy()
    
    contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        alpha_channel = np.zeros_like(alpha_channel)
        cv2.drawContours(alpha_channel, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    pad = 10
    padded_alpha = cv2.copyMakeBorder(alpha_channel, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    final_contours, _ = cv2.findContours(padded_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_pad, w_pad = padded_alpha.shape
    canvas = np.zeros((h_pad, w_pad, 4), dtype=np.uint8)
    cv2.drawContours(canvas, final_contours, -1, (0, 255, 0, 255), 3)
    
    final_contour_img = canvas[pad:-pad, pad:-pad]
    
    return final_contour_img