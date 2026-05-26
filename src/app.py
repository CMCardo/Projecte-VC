import streamlit as st
import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Trobem rutes automaticament
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR) 

# Carregar variables d'entorn (API Keys)
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("OPENTOPOGRAPHY_API_KEY")

# Assegurem que Python troba els nostres mòduls de src/
sys.path.append(CURRENT_DIR)
from segmentation import sky_remove_cv2, sky_remove_ai, sky_remove_laplace, sky_remove_hybrid
from profiling import get_1d_profile, draw_mountain_contour
from dem_engine import download_dynamic_dem, get_360_profile, plot_360_profile
from matching import find_best_match_correlation, find_best_match_mae, find_best_match_coincidence
from peaks_api import get_visible_peaks
from evaluation import extract_manual_profile, compare_contours
from utils import get_gps_from_photo

# Configuració de rutes
PHOTOS_DIR = os.path.join(BASE_DIR, "Photos")
GROUND_TRUTH_DIR = os.path.join(PHOTOS_DIR, "Ground_Truth")

st.set_page_config(page_title="Sky Remover App", layout="centered")
st.title("🏔️ Mountain Sky Remover & Peak Identifier")

# Seleccio d'imatge
st.markdown("### 1. Image Selection")
photo_num = st.number_input("Select photo number (1-44):", min_value=1, max_value=44, value=1)

image_path = None
extensions = [".jpg", ".JPG", ".jpeg", ".PNG", ".png"]
subcarpetes = ["Photos_Train", "Photos_Test", ""] # Busca a les subcarpetes

for sub in subcarpetes:
    for ext in extensions:
        ruta_temp = os.path.join(PHOTOS_DIR, sub, f"IMG_{photo_num}{ext}")
        if os.path.exists(ruta_temp):
            image_path = ruta_temp
            break
    if image_path:
        break

if not image_path:
    st.error(f"Image IMG_{photo_num} not found in Photos_Train or Photos_Test.")
    st.stop()

st.image(image_path, caption=f"Original Image (IMG_{photo_num})", use_container_width=True)

# Coordenades
st.markdown("---")
st.markdown("### 2. Camera Coordinates")
coords = get_gps_from_photo(image_path)
def_lat, def_lon, def_alt = coords if coords else (46.6358, 12.3164, 2405.0)

if not coords:
    st.info("GPS metadata not found. Using default coordinates (Tre Cime).!!!!")

col_lat, col_lon, col_alt = st.columns(3)
photo_lat = col_lat.number_input("Latitude:", value=float(def_lat), format="%.6f")
photo_lon = col_lon.number_input("Longitude:", value=float(def_lon), format="%.6f")
photo_alt = col_alt.number_input("Elevation (m):", value=float(def_alt), format="%.1f")

# Parametres de procesament
st.markdown("---")
st.markdown("### 3. Processing Settings")

# Settings DEM
col_dem1, col_dem2 = st.columns(2)
selected_dem = col_dem1.selectbox("Topographic Map Model:", ["COP30", "SRTMGL1", "COP90", "SRTMGL3"])
radius_km = col_dem2.slider("Map Download Radius (km):", 10, 80, 35, 5)

# Settings Segmentació
method = st.radio("Choose Sky Removal Method:", 
                  ("1. OpenCV Color (Blue Sky)", "2. AI (Deep Learning)", "3. Laplace (Texture)", "4. Hybrid (Color + Texture)"))

model_ai = "u2net"
tolerance = 3
if "2. AI" in method:
    model_ai = st.selectbox("AI Model:", ["u2net", "isnet-general-use", "u2netp", "silueta"])
elif "3. Laplace" in method or "4. Hybrid" in method:
    tolerance = st.slider("Texture Tolerance (Lower = keeps more texture):", 1, 50, 3)

draw_contour = st.checkbox("Show Mountain Border Line (Green)")

# Execucio
if st.button("Process Image & Find Peak", type="primary"):
    
    if not API_KEY:
        st.error("Missing OpenTopography API Key! Please add it to your .env file.")
        st.stop()

    with st.spinner("1/4 Removing Sky..."):
        if "1. OpenCV" in method:
            bgra_image = sky_remove_cv2(image_path)
        elif "2. AI" in method:
            bgra_image = sky_remove_ai(image_path, model_ai)
        elif "3. Laplace" in method:
            bgra_image = sky_remove_laplace(image_path, tolerance)
        else:
            bgra_image = sky_remove_hybrid(image_path, tolerance)

        if bgra_image is None:
            st.error("Segmentation failed.")
            st.stop()

        rgba_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA)
        st.image(rgba_image, caption="Sky Removed", use_container_width=True)

        if draw_contour:
            contour_img = draw_mountain_contour(bgra_image)
            if contour_img is not None:
                st.image(cv2.cvtColor(contour_img, cv2.COLOR_BGRA2RGBA), caption="Detected Silhouette", use_container_width=True)

    # Evaluacio amb ground truth
    photo_profile_1d = get_1d_profile(bgra_image)
    
    manual_route = None
    for ext in extensions:
        ruta_gt_temp = os.path.join(GROUND_TRUTH_DIR, f"IMG_{photo_num}{ext}")
        if os.path.exists(ruta_gt_temp):
            manual_route = ruta_gt_temp
            break
            
    if manual_route:
        st.markdown("### 📊 Border Precision Analysis (Ground Truth)")
        manual_profile = extract_manual_profile(manual_route)
        
        if manual_profile is not None:
            match_pct, error_pct, mean_dist = compare_contours(manual_profile, photo_profile_1d, tolerance_px=20)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Coincidence (±20px)", f"{match_pct:.1f} %")
            c2.metric("Error Rate", f"{error_pct:.1f} %")
            c3.metric("Mean Deviation", f"{mean_dist:.1f} px")
            
            fig_comp, ax_comp = plt.subplots(figsize=(10, 3))
            ax_comp.plot(photo_profile_1d, label="Algorithm Line (Green)", color='green')
            ax_comp.plot(manual_profile, label="Ground Truth (Red)", color='red', linestyle='dashed')
            ax_comp.legend()
            st.pyplot(fig_comp)
            plt.close(fig_comp) 
    
    # DEM i matching
    with st.spinner("2/4 Downloading/Loading 3D Topo Map (Cache)..."):
        dem_path = download_dynamic_dem(image_path, photo_lat, photo_lon, API_KEY, radius_km, selected_dem)
        
    if dem_path:
        with st.spinner("3/4 Calculating 360º Silhouette & Matching..."):
            dem_profile_360 = get_360_profile(dem_path, photo_lat, photo_lon, photo_alt, max_dist_km=radius_km)
            
            st.markdown("### Topographic Profile (360º DEM)")
            st.pyplot(plot_360_profile(dem_profile_360))
            
            ang_corr, fov_corr, sim_corr = find_best_match_correlation(photo_profile_1d, dem_profile_360)
            ang_mae, fov_mae, sim_mae = find_best_match_mae(photo_profile_1d, dem_profile_360)
            ang_coin, fov_coin, sim_coin = find_best_match_coincidence(photo_profile_1d, dem_profile_360, tolerance=0.05)

            st.markdown("### Alignment Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("1. Correlation (Robust)", f"{ang_corr}º Azimuth", f"{sim_corr*100:.1f}% Match")
            c2.metric("2. MAE Distance", f"{ang_mae}º Azimuth", f"{sim_mae:.1f}% Match")
            c3.metric("3. Coincidence Hits", f"{ang_coin}º Azimuth", f"{sim_coin:.1f}% Match")

        # Identificacio de pics
        with st.spinner(f"4/4 Searching Peaks in direction {ang_corr}º..."):
            st.markdown("---")
            st.markdown("### Identified Peaks")
            
            visible_peaks, total_zone, all_peaks = get_visible_peaks(photo_lat, photo_lon, ang_corr, fov_corr, radius_km)
            
            st.info(f"Radar scanned **{total_zone}** peaks in a {radius_km}km radius.")
            
            if visible_peaks:
                main_peak = visible_peaks[0]
                st.success(f"### **{main_peak['Peak Name']}**")
                colA, colB = st.columns(2)
                colA.metric("Elevation", f"{main_peak['Elevation (m)']} meters")
                colB.metric("Direction", f"{main_peak['Azimuth (º)']}º")
                
                if len(visible_peaks) > 1:
                    with st.expander(f"View other {len(visible_peaks)-1} peaks in the photo"):
                        st.dataframe(pd.DataFrame(visible_peaks[1:]), use_container_width=True)
            else:
                st.warning("No named peaks found in that specific direction according to OpenStreetMap.")
    else:
        st.error("DEM Map could not be loaded or downloaded.")