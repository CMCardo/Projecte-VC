import math
import requests

def calculate_bearing(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    
    initial_bearing = math.atan2(x, y)
    return (math.degrees(initial_bearing) + 360) % 360


def get_visible_peaks(observer_lat, observer_lon, angle_start, fov, radius_km=25):

    overpass_url = "https://lz4.overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node["natural"="peak"](around:{radius_km * 1000},{observer_lat},{observer_lon});
    out body;
    """
    headers = {'User-Agent': 'MountainSkyRemoverApp/2.0'}
    
    try:
        response = requests.get(overpass_url, params={'data': query}, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error connecting to OpenStreetMap: {e}")
        return [], 0, []

    elements = data.get('elements', [])
    total_peaks_in_zone = len(elements)
    
    angle_end = (angle_start + fov) % 360
    all_peaks = []

    for element in elements:
        if 'tags' in element and 'name' in element['tags']:
            peak_lat = element['lat']
            peak_lon = element['lon']
            
            bearing = calculate_bearing(observer_lat, observer_lon, peak_lat, peak_lon)
            
            # Comproba si un pic esta dins del fov de la camara.
            in_view = False
            if angle_start <= angle_end:
                if angle_start <= bearing <= angle_end:
                    in_view = True
            else: 
                # Cas per cuan el FOV pasa la marca de 360 graus.
                if bearing >= angle_start or bearing <= angle_end:
                    in_view = True
                    
            alt_str = element['tags'].get('ele', '0')
            try: 
                alt_float = float(alt_str)
            except ValueError: 
                alt_float = 0.0
                
            all_peaks.append({
                'Peak Name': element['tags']['name'],
                'Azimuth (º)': round(bearing, 1),
                'Elevation (m)': alt_float,
                'In Photo': '✅ Yes' if in_view else '❌ No'
            })

    # Ordenem per elevació
    all_peaks.sort(key=lambda x: x['Elevation (m)'], reverse=True)
    visible_peaks = [peak for peak in all_peaks if peak['In Photo'] == '✅ Yes']
    
    return visible_peaks, total_peaks_in_zone, all_peaks
