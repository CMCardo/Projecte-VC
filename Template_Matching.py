import numpy as np
import cv2

def find_best_match(photo_border, dem_profile, min_fov=30, max_fov=100):
    
    best_corr = -1.0
    best_fov = 0
    best_angle = 0
    
    #Extent the DEM to avoid errors
    dem_extended = np.concatenate((dem_profile, dem_profile))
    
    #opencv needs 2d arrays, add 1 dimension
    dem_2d = dem_extended.reshape(1, -1).astype(np.float32)
    foto_1d = np.array(photo_border, dtype=np.float32)
    
    for fov in range(min_fov, max_fov + 1):
        
        photo_resized_2d = cv2.resize(foto_1d.reshape(1, -1), (fov, 1), interpolation=cv2.INTER_LINEAR)
        
        #correlation, TM_CCOEFF_NORMED ignores if the photo is in pixels or degrees, only in ups and downs
        result = cv2.matchTemplate(dem_2d, photo_resized_2d, cv2.TM_CCOEFF_NORMED)
        
        #higher value
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_corr:
            best_corr = max_val
            best_fov = fov
            best_angle = max_loc[0] 
            
    best_angle = best_angle % 360
            
    return best_angle, best_fov, best_corr


def prepare_photo_profile(image_path):

    # read image with all channels
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]
    
    border_1d = np.zeros(width)
    
    #check if transparency
    if img.shape[2] == 4:
        mask = img[:, :, 3] # take only transparency
    else:
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # if has no transparency, convert to grey
    
    # column by column till the first solid pixel
    for x in range(width):
        y_mountain = np.argmax(mask[:, x] > 10) #pixel value > 10
        
        if y_mountain == 0 and mask[0, x] <= 10:
            border_1d[x] = 0 # column is sky
        else:
            border_1d[x] = height - y_mountain #real height
            
    #aply blur, delete noise
    border_1d = cv2.GaussianBlur(border_1d.astype(np.float32).reshape(1, -1), (15, 1), 0)[0]
            
    return border_1d



def find_best_match_points(photo_border, dem_profile, min_fov=30, max_fov=100):

    best_error = float('inf')
    best_fov = 0
    best_angle = 0
    
    dem_extended = np.concatenate((dem_profile, dem_profile))
    
    # normalize the photo
    photo_norm = (photo_border - np.min(photo_border)) / (np.max(photo_border) - np.min(photo_border) + 1e-5)

    for fov in range(min_fov, max_fov + 1):
        
        # Resize normalized photo
        photo_resized = cv2.resize(photo_norm.reshape(1, -1), (fov, 1), interpolation=cv2.INTER_LINEAR)[0]
        
        #slide this photo through each corner of the DEM
        for angle in range(360):
            
            dem_window = dem_extended[angle : angle + fov] #cut the current part of DEM
            dem_window_norm = (dem_window - np.min(dem_window)) / (np.max(dem_window) - np.min(dem_window) + 1e-5) #normalize
            
            current_error = np.mean(np.abs(photo_resized - dem_window_norm))
            
            if current_error < best_error:
                best_error = current_error
                best_angle = angle
                best_fov = fov
                
    best_angle = best_angle % 360
    
    correct_perc = (1.0 - best_error) * 100
                
    return best_angle, best_fov, correct_perc



def find_best_match_coincidence(photo_border, dem_profile, min_fov=30, max_fov=100, tolerance=0.05):

    best_score = -1.0 
    best_fov = 0
    best_angle = 0
    
    dem_extended = np.concatenate((dem_profile, dem_profile))
    photo_norm = (photo_border - np.min(photo_border)) / (np.max(photo_border) - np.min(photo_border) + 1e-5)
    
    for fov in range(min_fov, max_fov + 1):
        
        photo_resized = cv2.resize(photo_norm.reshape(1, -1), (fov, 1), interpolation=cv2.INTER_LINEAR)[0]
        
        for angle in range(360):

            dem_window = dem_extended[angle : angle + fov]
            dem_window_norm = (dem_window - np.min(dem_window)) / (np.max(dem_window) - np.min(dem_window) + 1e-5)
            
            # calculate the exact distance between each pair of points
            diff = np.abs(photo_resized - dem_window_norm)
            
            # how many points have a distance smaller than the tolerance
            coincident_points = np.sum(diff < tolerance)
            
            # % of lines match
            score_actual = (coincident_points / fov) * 100
            
            if score_actual > best_score:
                best_score = score_actual
                best_angle = angle
                best_fov = fov
                
    best_angle = best_angle % 360
                
    return best_angle, best_fov, best_score




def extract_manual_profile(image_path):

    img = cv2.imread(image_path)
    if img is None:
        return None
        
    #detect red
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    height, width = red_mask.shape
    profile = np.full(width, np.nan)
    
    #search for first red pixel in column 
    for x in range(width):
        y_indices = np.where(red_mask[:, x] > 0)[0]
        if len(y_indices) > 0:
            profile[x] = height - y_indices[0]
            
    return profile



def compare_contours(profile_manual, profile_algo, tolerance_px=5):

    if profile_manual is None or profile_algo is None:
        return 0.0, 0.0, 0.0
        
    min_len = min(len(profile_manual), len(profile_algo))
    p_man = profile_manual[:min_len]
    p_alg = profile_algo[:min_len]
    
    valid_cols = ~np.isnan(p_man) & ~np.isnan(p_alg)
    
    p_man_valid = p_man[valid_cols]
    p_alg_valid = p_alg[valid_cols]
    
    if len(p_man_valid) == 0:
        return 0.0, 0.0, 0.0
        
    diffs = np.abs(p_man_valid - p_alg_valid)
    
    matches = np.sum(diffs <= tolerance_px)
    errors = len(diffs) - matches
    
    match_pct = (matches / len(diffs)) * 100
    error_pct = (errors / len(diffs)) * 100
    mean_dist = np.mean(diffs)
    
    return match_pct, error_pct, mean_dist