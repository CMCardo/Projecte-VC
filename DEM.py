import os
import requests
import numpy as np
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt

def get_360_profile(dem_path, lat, lon, user_alt, max_dist_km=30):

    with rasterio.open(dem_path) as src:
        dem_matrix = src.read(1)
        
        # Transform gps coord to DEM
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x_crs, y_crs = transformer.transform(lon, lat)
        
        row, col = src.index(x_crs, y_crs)
        
        # Pixel size in m
        pixel_size_m = 90
            
        max_steps = int((max_dist_km * 1000) / pixel_size_m)
        profile_360 = np.zeros(360)
        
        # Ray-casting
        for angle in range(360):
            rad = np.radians(angle)
            max_theta = -np.inf
            
            sin_a = np.sin(rad)
            cos_a = np.cos(rad)
            
            for step in range(1, max_steps):
                target_col = int(col + step * sin_a)
                target_row = int(row - step * cos_a) 
                
                if 0 <= target_row < dem_matrix.shape[0] and 0 <= target_col < dem_matrix.shape[1]:
                    altitud_terreny = dem_matrix[target_row, target_col]
                    
                    if altitud_terreny == src.nodata or altitud_terreny < -500:
                        continue
                        
                    distancia_m = step * pixel_size_m
                    theta = np.degrees(np.arctan2((altitud_terreny - user_alt), distancia_m))
                    
                    if theta > max_theta:
                        max_theta = theta
                else:
                    break 
                    
            profile_360[angle] = max_theta if max_theta != -np.inf else 0
            
    return profile_360



def plot_360_profile(profile_360):

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(profile_360, color='#1f77b4', linewidth=2.5, label="DEM siluete")
    ax.fill_between(range(360), profile_360, color='#1f77b4', alpha=0.15)
    
    ax.set_xlabel("Orientation", fontsize=11)
    ax.set_ylabel("elevation Angle", fontsize=11)
    
    ax.set_xlim(0, 359)
    ax.set_xticks([0, 90, 180, 270, 359])
    ax.set_xticklabels(['N (0º)', 'E (90º)', 'S (180º)', 'O (270º)', 'N (360º)'])
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig


def get_user_altitude_from_dem(dem_path, lat, lon):

    with rasterio.open(dem_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x_crs, y_crs = transformer.transform(lon, lat)
        
        coordenades_projectades = [(x_crs, y_crs)]
        generator = src.sample(coordenades_projectades)
        
        for valors_píxel in generator:
            altitud = valors_píxel[0]
            return altitud


def download_dynamic_dem(lat, lon, api_key, radi_km=15):
    
    graus_radi = radi_km / 111.0
    
    south = lat - graus_radi
    north = lat + graus_radi
    west = lon - graus_radi
    east = lon + graus_radi

    url = f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL3&south={south}&north={north}&west={west}&east={east}&outputFormat=GTiff&API_Key={api_key}"
    
    resposta = requests.get(url, stream=True)
    
    if resposta.status_code == 200:
        temp_filename = "temp_dem.tif"
        with open(temp_filename, 'wb') as f:
            for chunk in resposta.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Mapa downloaded")
        return temp_filename
    else:
        print(f"Error downloading map: {resposta.status_code}")
        print(f"Reason: {resposta.text}") 
        return None