import os
import requests
import numpy as np
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt
import math
import requests

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


def download_dynamic_dem(lat, lon, api_key, radi_km=35, dem_type="COP90"):
    
    graus_radi = radi_km / 111.0
    
    south = lat - graus_radi
    north = lat + graus_radi
    west = lon - graus_radi
    east = lon + graus_radi

   # MODIFIED: demtype is now a dynamic variable
    url = f"https://portal.opentopography.org/API/globaldem?demtype={dem_type}&south={south}&north={north}&west={west}&east={east}&outputFormat=GTiff&API_Key={api_key}"
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
    



def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calcula l'angle (azimut) des del punt 1 fins al punt 2."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    initial_bearing = math.atan2(x, y)
    
    # Passem de radians a graus i assegurem que estigui entre 0 i 360
    return (math.degrees(initial_bearing) + 360) % 360


def get_visible_peaks(observer_lat, observer_lon, angle_start, fov, radius_km=25):
    """
    Busca cims a OpenStreetMap i retorna la llista de cims visibles, 
    el total, i la llista absoluta de tots els cims amb el seu estat.
    """
    import requests
    import math
    
    overpass_url = "https://lz4.overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node["natural"="peak"](around:{radius_km * 1000},{observer_lat},{observer_lon});
    out body;
    """
    
    headers = {'User-Agent': 'MountainSkyRemoverApp/1.0'}
    
    try:
        response = requests.get(overpass_url, params={'data': query}, headers=headers, timeout=10)
        data = response.json()
    except Exception as e:
        print(f"Error connectant amb OpenStreetMap: {e}")
        return [], 0, []

    elements = data.get('elements', [])
    total_cims_zona = len(elements)
    
    angle_end = (angle_start + fov) % 360
    tots_els_cims = []

    for element in elements:
        if 'tags' in element and 'name' in element['tags']:
            peak_lat = element['lat']
            peak_lon = element['lon']
            
            bearing = calculate_bearing(observer_lat, observer_lon, peak_lat, peak_lon)
            
            in_view = False
            if angle_start < angle_end:
                if angle_start <= bearing <= angle_end:
                    in_view = True
            else: 
                if bearing >= angle_start or bearing <= angle_end:
                    in_view = True
                    
            alt = element['tags'].get('ele', '0')
            try: 
                alt_float = float(alt)
            except: 
                alt_float = 0.0
                
            tots_els_cims.append({
                'Nom del Cim': element['tags']['name'],
                'Azimut (º)': round(bearing, 1),
                'Altitud (m)': alt_float,
                'Dins la Foto?': '✅ Sí' if in_view else '❌ No'
            })

    # Ordenem per altitud
    tots_els_cims.sort(key=lambda x: x['Altitud (m)'], reverse=True)
    
    # Filtrem només els visibles per a la taula principal
    cims_visibles = [cim for cim in tots_els_cims if cim['Dins la Foto?'] == '✅ Sí']
    
    # FIX: Retornem exactament les 3 coses que demana l'app.py
    return cims_visibles, total_cims_zona, tots_els_cims