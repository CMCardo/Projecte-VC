import numpy as np
import cv2

def find_best_match_correlation(photo_profile, dem_profile, min_fov=30, max_fov=100):
    best_corr = -1.0
    best_fov = 0
    best_angle = 0
    
    dem_extended = np.concatenate((dem_profile, dem_profile))

    # Esto soluciona por completo el problema de los valles largos.
    dem_diff = np.gradient(dem_extended)
    photo_diff = np.gradient(photo_profile)
    
    dem_2d = dem_diff.reshape(1, -1).astype(np.float32)
    photo_1d = np.array(photo_diff, dtype=np.float32)
    
    for fov in range(min_fov, max_fov + 1):
        photo_resized_2d = cv2.resize(photo_1d.reshape(1, -1), (fov, 1), interpolation=cv2.INTER_LINEAR)
        
        result = cv2.matchTemplate(dem_2d, photo_resized_2d, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_corr:
            best_corr = max_val
            best_fov = fov
            best_angle = max_loc[0] 
            
    return best_angle % 360, best_fov, best_corr


def find_best_match_mae(photo_profile, dem_profile, min_fov=30, max_fov=100):
    """
    Metode 2: Media del valor absolut
    """
    best_error = float('inf')
    best_fov = 0
    best_angle = 0
    
    dem_extended = np.concatenate((dem_profile, dem_profile))
    
    photo_range = np.max(photo_profile) - np.min(photo_profile)
    photo_norm = (photo_profile - np.min(photo_profile)) / (photo_range + 1e-8)

    for fov in range(min_fov, max_fov + 1):
        photo_resized = cv2.resize(photo_norm.reshape(1, -1), (fov, 1), interpolation=cv2.INTER_LINEAR)[0]
        
        for angle in range(360):
            dem_window = dem_extended[angle : angle + fov] 
            
            dem_range = np.max(dem_window) - np.min(dem_window)
            if dem_range < 0.1: # It's a flat surface, don't stretch it
                dem_window_norm = np.zeros_like(dem_window)
            else:
                dem_window_norm = (dem_window - np.min(dem_window)) / dem_range
            
            current_error = np.mean(np.abs(photo_resized - dem_window_norm))
            
            if current_error < best_error:
                best_error = current_error
                best_angle = angle
                best_fov = fov
                
    best_angle = best_angle % 360
    
    # Convertim l'error a un percentatge de similitud
    similarity_perc = max(0.0, (1.0 - best_error) * 100)
                
    return best_angle, best_fov, similarity_perc


def find_best_match_coincidence(photo_profile, dem_profile, min_fov=30, max_fov=100, tolerance=0.05):

    best_score = -1.0 
    best_fov = 0
    best_angle = 0
    
    dem_extended = np.concatenate((dem_profile, dem_profile))
    
    photo_range = np.max(photo_profile) - np.min(photo_profile)
    photo_norm = (photo_profile - np.min(photo_profile)) / (photo_range + 1e-8)
    
    for fov in range(min_fov, max_fov + 1):
        photo_resized = cv2.resize(photo_norm.reshape(1, -1), (fov, 1), interpolation=cv2.INTER_LINEAR)[0]
        
        for angle in range(360):
            dem_window = dem_extended[angle : angle + fov]
            
            dem_range = np.max(dem_window) - np.min(dem_window)
            if dem_range < 0.1:
                dem_window_norm = np.zeros_like(dem_window)
            else:
                dem_window_norm = (dem_window - np.min(dem_window)) / dem_range
            
            # Distancia normalitzada entre punts
            diff = np.abs(photo_resized - dem_window_norm)
            
            # Contem el nomre de punts que estan mes aprop que la tolerancia
            coincident_points = np.sum(diff < tolerance)
            
            # Calculem el percentatge de hit
            score_actual = (coincident_points / fov) * 100
            
            if score_actual > best_score:
                best_score = score_actual
                best_angle = angle
                best_fov = fov
                
    best_angle = best_angle % 360
                
    return best_angle, best_fov, best_score
