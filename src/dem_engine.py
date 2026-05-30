import os
import requests
import numpy as np
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
CACHE_DIR = os.path.join(BASE_DIR, "Cache")

def get_photo_name_from_path(image_path):
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    return name_without_ext

def download_dynamic_dem(image_path, lat, lon, api_key, radius_km=35, dem_type="COP90"):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    photo_name = get_photo_name_from_path(image_path)
    cache_filename = os.path.join(CACHE_DIR, f"DEM_{photo_name}_{dem_type}_{radius_km}km.tif")

    # En cas de que ja tinguem en cache el dem de la foto, el fem servir
    if os.path.exists(cache_filename):
        print(f"Loaded DEM from cache: {cache_filename}")
        return cache_filename

    # En cas de que no tinguem en cache, fem servir la api
    radius_degree = radius_km / 111.0
    south = lat - radius_degree
    north = lat + radius_degree
    west = lon - radius_degree
    east = lon + radius_degree

    url = f"https://portal.opentopography.org/API/globaldem?demtype={dem_type}&south={south}&north={north}&west={west}&east={east}&outputFormat=GTiff&API_Key={api_key}"
    
    print(f"Downloading DEM for {photo_name} from OpenTopography...")
    res = requests.get(url, stream=True)
    
    if res.status_code == 200:
        with open(cache_filename, 'wb') as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Map successfully downloaded and cached at: {cache_filename}")
        return cache_filename
    else:
        print(f"Error downloading map: {res.status_code}")
        print(f"Reason: {res.text}") 
        return None

def get_360_profile(dem_path, lat, lon, user_alt, max_dist_km=30):

    with rasterio.open(dem_path) as src:
        dem_matrix = src.read(1)
        
        # Transformar coordenades gps a la referencia en un mapa DEM
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x_crs, y_crs = transformer.transform(lon, lat)
        
        row, col = src.index(x_crs, y_crs)
        
        # Hem de tenir en compte la altura dels pixels en metres
        pixel_size_m = 90
        max_steps = int((max_dist_km * 1000) / pixel_size_m)
        profile_360 = np.zeros(360)
        
        # Bucle de raycasting per a cada grau
        for angle in range(360):
            rad = np.radians(angle)
            max_theta = -np.inf
            
            sin_a = np.sin(rad)
            cos_a = np.cos(rad)
            
            for step in range(1, max_steps):
                target_col = int(col + step * sin_a)
                target_row = int(row - step * cos_a) 
                
                if 0 <= target_row < dem_matrix.shape[0] and 0 <= target_col < dem_matrix.shape[1]:
                    terrain_altitude = dem_matrix[target_row, target_col]
                    
                    if terrain_altitude == src.nodata or terrain_altitude < -500:
                        continue
                        
                    distance_m = step * pixel_size_m
                    # Calcular angle d'elevació
                    theta = np.degrees(np.arctan2((terrain_altitude - user_alt), distance_m))
                    
                    if theta > max_theta:
                        max_theta = theta
                else:
                    break 
                    
            profile_360[angle] = max_theta if max_theta != -np.inf else 0
            
    return profile_360

def plot_360_profile(profile_360):

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(profile_360, color='#1f77b4', linewidth=2.5, label="DEM Silhouette")
    ax.fill_between(range(360), profile_360, color='#1f77b4', alpha=0.15)
    
    ax.set_xlabel("Orientation (Azimuth)", fontsize=11)
    ax.set_ylabel("Elevation Angle (Degrees)", fontsize=11)
    
    ax.set_xlim(0, 359)
    ax.set_xticks([0, 90, 180, 270, 359])
    ax.set_xticklabels(['N (0º)', 'E (90º)', 'S (180º)', 'W (270º)', 'N (360º)'])
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig
