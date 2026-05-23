import streamlit as st
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

from photo_loader import sky_remove_cv2, sky_remove_ai, sky_remove_Laplace, sky_remove_hybrid, get_gps_from_photo
from functions import extract_mountain_contour
from DEM import get_360_profile, plot_360_profile, download_dynamic_dem, get_visible_peaks
from Template_Matching import prepare_photo_profile, find_best_match, find_best_match_points, find_best_match_coincidence, find_best_match_coincidence, extract_manual_profile, compare_contours

MAX_NUM_PHOTOS = 44
API_KEY = "8cfe74cb60bee3b3fff0d215fdb1bcda"


# Page configuration
st.set_page_config(page_title="Sky Remover App", layout="centered")

st.title("Mountain Sky Remover")
st.write("Select a photo number and the processing method to remove the sky.")

# Select photo number
photo_num = st.number_input(f"Select photo number (1-{MAX_NUM_PHOTOS}):", min_value=1, max_value=MAX_NUM_PHOTOS, value=1)

# Display the original image
image_name = None
extensions_suportades = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".webp", ".WEBP"] #extensions

for ext in extensions_suportades:
    ruta_temporal = f"Photos/IMG_{photo_num}{ext}"
    if os.path.exists(ruta_temporal):
        image_name = ruta_temporal
        break  

if image_name is None:
    image_name = f"Photos/IMG_{photo_num}.JPG"

if os.path.exists(image_name):
    st.image(image_name, caption=f"Original Image (IMG_{photo_num})", use_container_width=True)
else:
    st.error(f"Image not found at {image_name}. Please check your 'Photos' folder.")

st.markdown("---")

# coords selector
st.subheader("Coordinates of the picture:")

# Take gps metadata
coords_auto = get_gps_from_photo(image_name) if os.path.exists(image_name) else None

if coords_auto:
    def_lat, def_lon, def_alt = coords_auto
else:
    # Default values, tre cime
    def_lat, def_lon, def_alt = 46.6358, 12.3164, 2405.0
    st.info("GPS coordinates not found on the foto, using default coordinates.")

col_lat, col_lon, col_alt = st.columns(3)
photo_lat = col_lat.number_input("Latitude:", value=float(def_lat), format="%.6f")
photo_lon = col_lon.number_input("Length:", value=float(def_lon), format="%.6f")
foto_alt = col_alt.number_input("Elevation (metres):", value=float(def_alt), format="%.1f")

st.markdown("---")

# DEM
st.subheader("Topografic Map and Radius")
dem_options = {
    "NASA SRTMGL3 (Use between Lat 60º North and 56º South)": "SRTMGL3",
    "Copernicus COP90 (Global)": "COP90"
}
selected_dem_label = st.selectbox("Select which satelite model you want to use:", list(dem_options.keys()))
selected_dem = dem_options[selected_dem_label]

# DEM km radius selector
radius_km_usuari = st.slider("Analisis and download radius of the map (km):", min_value=10, max_value=80, value=35, step=5)

# Tolerance of the contourn error in pixels
tolerance_contorn = st.slider("Tolerance of the contourn error in pixels:", min_value=1, max_value=50, value=20, step=1)

st.markdown("---")


# Select processing method
method = st.radio("Choose the processing method:", 
                  ("1. OpenCV Color (Pure blue skies)", 
                   "2. AI", 
                   "3. Laplace (Texture detection)",
                   "4. Hybrid (Color + Texture)"))

# Select AI Model
model = "u2net" # Default 
tolerance = 3

if method == "2. AI":
    ai_models = {
        "isnet-general-use (Most complex)": "isnet-general-use",
        "u2net (Default)": "u2net",
        "u2netp": "u2netp",
        "silueta (Least complex)": "silueta"
    }
    selected_model_label = st.selectbox("Select AI Model:", list(ai_models.keys()))
    model = ai_models[selected_model_label]

elif method == "3. Laplace (Texture detection)":
    st.info("Lower tolerance = keeps more texture. Higher tolerance = removes more smooth parts.")
    # Slider for tolerance
    tolerance = st.slider("Select Texture Tolerance:", min_value=1, max_value=50, value=3)

elif method == "4. Hybrid (Color + Texture)":
    tolerance = st.slider("Select Texture Tolerance for Hybrid:", 1, 50, 3)

# Checkbox for border
draw_contour = st.checkbox("Show mountain border")

# Process execution
if st.button("Process Image"):
    if not os.path.exists(image_name):
         st.error("Cannot process because the original image is missing.")
    else:
        with st.spinner("Processing image and downloading 3D map ... Please wait."):
            
            result_path = None

            if method == "1. OpenCV Color (Pure blue skies)":
                sky_remove_cv2(photo_num)
                result_path = f"sky_remove_cv2/mountain_{photo_num}_color.png"
                st.success("Sky removed successfully using OpenCV!")
                
            elif method == "2. AI":
                sky_remove_ai(photo_num, model)
                result_path = f"sky_remove_ai/mountain_{photo_num}_cutted.png"
                st.success(f"Processed with AI ({model})!")
                
            elif method == "3. Laplace (Texture detection)":
                result_path = sky_remove_Laplace(photo_num, tolerance)
                st.success(f"Processed with Laplace (Tolerance: {tolerance})!")

            elif method == "4. Hybrid (Color + Texture)": 
                result_path = sky_remove_hybrid(photo_num, tolerance)
                st.success("Hybrid mode applied: Only pixels that are both BLUE and SMOOTH were removed.")
        
            
            if result_path and os.path.exists(result_path):
                st.image(result_path, caption="Final Result", use_container_width=True)

                if draw_contour:
                    line_rute = extract_mountain_contour(result_path)
                    if line_rute:
                        st.image(line_rute, caption="Mountain line", use_container_width=True)

            # Precision analisis
                manual_route = f"border_test_photo/IMG_{photo_num}.jpg"
                
                if os.path.exists(manual_route):
                    st.markdown("---")
                    st.subheader("Border precision analsis")
                    
                    with st.spinner("Comparing algorith result with Ground Truth..."):
                        
                        manual_border = extract_manual_profile(manual_route)
                        algo_border = prepare_photo_profile(result_path)
                        
                        encert_pct, error_pct, dist_mitjana = compare_contours(manual_border, algo_border, tolerance_px=tolerance_contorn)
                        col1, col2, col3 = st.columns(3)
                        
                        # Analisis result
                        col1.metric(f"Coincidence (±{tolerance_contorn}px)", f"{encert_pct:.1f} %")
                        col2.metric("Error (Outside line)", f"{error_pct:.1f} %")
                        col3.metric("Desviació mitjana", f"{dist_mitjana:.1f} píxels")
                        fig_comp, ax_comp = plt.subplots(figsize=(10, 3.5))
                        
                        # Drawing the 2 lines
                        ax_comp.plot(algo_border, label="Algortithm line (Green)", color='green', linewidth=2)
                        ax_comp.plot(manual_border, label="Ground Truth (Red)", color='red', linewidth=2, linestyle='dashed')
                        
                        ax_comp.set_title("Ground Truth vs Algorisme")
                        ax_comp.set_xlabel("Width")
                        ax_comp.set_ylabel("Height")
                        ax_comp.legend()
                        st.pyplot(fig_comp)
                        
                else:
                    st.info(f"Ground Truth for `{manual_route}` not found.")
                
                st.markdown("---")
                st.header("Topografic Analisis")

                # download the map
                dem_temporal = download_dynamic_dem(photo_lat, photo_lon, api_key=API_KEY, radius_km=radius_km_usuari, dem_type=selected_dem)

                if dem_temporal:
                    
                    virtual_border = get_360_profile(dem_temporal, photo_lat, photo_lon, foto_alt, max_dist_km=radius_km_usuari)
                    st.pyplot(plot_360_profile(virtual_border))
                    os.remove(dem_temporal)

                    photo_array_1d = prepare_photo_profile(result_path)
                    
                    fig2, ax2 = plt.subplots(figsize=(8, 3))
                    ax2.plot(photo_array_1d, color='green', linewidth=2)
                    ax2.set_title("1D border from the photo")
                    st.pyplot(fig2)

                    angle_corr, fov_corr, sim_corr = find_best_match(photo_array_1d, virtual_border)
                    angle_pts, fov_pts, sim_pts = find_best_match_points(photo_array_1d, virtual_border)
                    angle_hits, fov_hits, score_hits = find_best_match_coincidence(photo_array_1d, virtual_border, tolerance=0.05)

                    # show result
                    st.subheader("Alignment results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Correlation methot", f"{angle_corr}º", f"{sim_corr*100:.1f}% certesa")
                    col2.metric("Point per point methot", f"{angle_pts}º", f"{sim_pts:.1f}% certesa")
                    col3.metric("Coincidence methot", f"{angle_hits}º", f"{score_hits:.1f}% Hits")


                    st.markdown("---")
                    st.header("Match Visualization per Method")

                    def plot_basic_match(angle, fov, title, color_zone):
                        fig, ax_dem = plt.subplots(figsize=(10, 3.5))
                        
                        ax_dem.plot(virtual_border, color='#1f77b4', linewidth=2)
                        ax_dem.fill_between(range(360), virtual_border, color='#1f77b4', alpha=0.15)
                        ax_dem.set_xlim(0, max(359, angle + fov))
                        ax_dem.set_xlabel("Azimut (Graus)")
                        ax_dem.set_ylabel("Elevació DEM", color='#1f77b4')
                        ax_dem.axvspan(angle, angle + fov, color=color_zone, alpha=0.2, label="Zona d'Encaix")
                        
                        ax_foto = ax_dem.twinx()
                        foto_pixels_resized = cv2.resize(photo_array_1d.astype(np.float32).reshape(1, -1), (fov, 1), interpolation=cv2.INTER_LINEAR)[0]
                        eix_x = np.arange(angle, angle + fov)
                        
                        ax_foto.plot(eix_x, foto_pixels_resized, color='green', linewidth=2, alpha=0.5, label="Photo border")
                        ax_foto.set_ylabel("Photo pixels", color='green')
                        
                        plt.title(title, fontweight='bold')
                        return fig, ax_foto, foto_pixels_resized, eix_x

                    #correlation graph
                    fig_corr, _, _, _ = plot_basic_match(angle_corr, fov_corr, f"1. Correlation (Azimut: {angle_corr}º | Zoom: {fov_corr}º)", 'orange')
                    st.pyplot(fig_corr)

                    # point per point graph
                    fig_pts, _, _, _ = plot_basic_match(angle_pts, fov_pts, f"2. Point per point distance (Azimut: {angle_pts}º | Zoom: {fov_pts}º)", 'purple')
                    st.pyplot(fig_pts)

                    # coincidence graph
                    fig_hits, ax_foto, foto_pixels_resized, eix_x = plot_basic_match(angle_hits, fov_hits, f"3. Coincidence of points (Azimut: {angle_hits}º | Hits: {score_hits:.1f}%)", 'green')
                    
                    dem_extended = np.concatenate((virtual_border, virtual_border))
                    dem_window = dem_extended[angle_hits : angle_hits + fov_hits]
                    
                    foto_norm = (photo_array_1d - np.min(photo_array_1d)) / (np.max(photo_array_1d) - np.min(photo_array_1d) + 1e-5)
                    foto_res_norm = cv2.resize(foto_norm.reshape(1, -1), (fov_hits, 1), interpolation=cv2.INTER_LINEAR)[0]
                    dem_window_norm = (dem_window - np.min(dem_window)) / (np.max(dem_window) - np.min(dem_window) + 1e-5)
                    
                    tolerance = 0.05
                    mask_hits = np.abs(foto_res_norm - dem_window_norm) < tolerance
                    
                    ax_foto.scatter(eix_x[mask_hits], foto_pixels_resized[mask_hits], color='lime', s=25, zorder=5, label='Coincidence (YES)')
                    ax_foto.scatter(eix_x[~mask_hits], foto_pixels_resized[~mask_hits], color='red', s=25, zorder=5, label='Error (NO)')
                    ax_foto.legend(loc='upper right')
                    
                    st.pyplot(fig_hits)

                    # peak identificator
                    st.markdown("---")
                    st.header("Main peak identificator")
                    
                    with st.spinner("Cross-referencing coordinates with the OpenStreetMap database..."):
                        
                        visible_peaks, total_zone, all_the_peaks = get_visible_peaks(photo_lat, photo_lon, angle_hits, fov_hits, radius_km=radius_km_usuari)
                        
                        st.info(f"Radar: **{total_zone}** peaks have been detected within a radius of  {radius_km_usuari}km.")
                        
                        if visible_peaks:
                            # most probable peak
                            main_peak = visible_peaks[0]
                            
                            st.markdown(f"### **{main_peak['Peak name']}**")
                            colA, colB = st.columns(2)
                            colA.metric("Elevation", f"{main_peak['Elevation (m)']} meters")
                            colB.metric("Photo Azimut", f"{main_peak['Azimut (º)']}º")
                            
                            st.write("This is the most relevant peak identified right in the direction your photo is pointing.")
                            
                            # other peaks
                            if len(visible_peaks) > 1:
                                with st.expander(f"See other {len(visible_peaks) - 1} peaks"):
                                    df_secundaris = pd.DataFrame(visible_peaks[1:])
                                    st.dataframe(df_secundaris[['Peak name', 'Azimut (º)', 'Elevation (m)']], use_container_width=True)
                                    
                        else:
                            st.warning("The algorithm is pointing towards an area where there are no visible peaks.")
                            
                        # all the peak in the DEM
                        if all_the_peaks:
                            with st.expander(" See all the peaks in the DEM"):
                                st.dataframe(pd.DataFrame(all_the_peaks), use_container_width=True)

                else:
                    st.error("Could not connect to 3D map.")
            else:
                st.error("Something went wrong. The resulting image was not saved properly.")