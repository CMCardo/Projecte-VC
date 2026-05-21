import streamlit as st
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

from photo_loader import sky_remove_cv2, sky_remove_ai, sky_remove_Laplace, sky_remove_hybrid
from functions import extract_mountain_contour
from DEM import get_360_profile, plot_360_profile, download_dynamic_dem
from Template_Matching import prepare_photo_profile, find_best_match, find_best_match_points, find_best_match_coincidence

MAX_NUM_PHOTOS = 43

# Page configuration
st.set_page_config(page_title="Sky Remover App", layout="centered")

st.title("Mountain Sky Remover")
st.write("Select a photo number and the processing method to remove the sky.")

# Select photo number
photo_num = st.number_input(f"Select photo number (1-{MAX_NUM_PHOTOS}):", min_value=1, max_value=MAX_NUM_PHOTOS, value=1)

# Display the original image
image_name = f"Photos/IMG_{photo_num}.JPG"
if not os.path.exists(image_name):
    # Check for lowercase extension
    image_name = f"Photos/IMG_{photo_num}.jpg"

if os.path.exists(image_name):
    st.image(image_name, caption=f"Original Image (IMG_{photo_num})", use_container_width=True)
else:
    st.error(f"Image not found at {image_name}. Please check your 'Photos' folder.")

st.markdown("---")

# Select processing method
method = st.radio("Choose the processing method:", 
                  ("1. OpenCV Color (Pure blue skies)", 
                   "2. AI (Complex landscapes)", 
                   "3. Laplace (Texture detection)",
                   "4. Hybrid (Color + Texture)"))

# Select AI Model
model = "u2net" # Default 
tolerance = 3

if method == "2. AI (Complex landscapes)":
    ai_models = {
        "isnet-general-use (Best for landscapes and nature)": "isnet-general-use",
        "u2net (Default, good for general objects)": "u2net",
        "u2netp (Lightweight and faster version)": "u2netp",
        "silueta (Extremely small and fast model)": "silueta"
    }
    selected_model_label = st.selectbox("Select AI Model:", list(ai_models.keys()))
    model = ai_models[selected_model_label]

elif method == "3. Laplace (Texture detection)":
    st.info("Lower tolerance = keeps more texture. Higher tolerance = removes more 'smooth' parts.")
    # Slider for tolerance
    tolerance = st.slider("Select Texture Tolerance:", min_value=1, max_value=50, value=3)

elif method == "4. Hybrid (Color + Texture)":
    tolerance = st.slider("Select Texture Tolerance for Hybrid:", 1, 50, 3)

# Checkbox for border
draw_contour = st.checkbox("Only mountain border (Siluet)")

# Process execution
if st.button("Process Image"):
    if not os.path.exists(image_name):
         st.error("Cannot process because the original image is missing.")
    else:
        with st.spinner("Processing image i descarregant mapa 3D... Please wait."):
            
            result_path = None

            if method == "1. OpenCV Color (Pure blue skies)":
                sky_remove_cv2(photo_num)
                result_path = f"sky_remove_cv2/mountain_{photo_num}_color.png"
                st.success("Sky removed successfully using OpenCV!")
                
            elif method == "2. AI (Complex landscapes)":
                sky_remove_ai(photo_num, model)
                result_path = f"sky_remove_ai/mountain_{photo_num}_cutted.png"
                st.success(f"Processed with AI ({model})!")
                
            elif method == "3. Laplace (Texture detection)":
                result_path = sky_remove_Laplace(photo_num, tolerance)
                st.success(f"Processed with Laplace (Tolerance: {tolerance})!")

            elif method == "4. Hybrid (Color + Texture)": 
                result_path = sky_remove_hybrid(photo_num, tolerance)
                st.success("Hybrid mode applied: Only pixels that are both BLUE and SMOOTH were removed.")
        
            
            # --- COMPROVEM SI LA FOTO S'HA GENERAT BÉ ---
            if result_path and os.path.exists(result_path):
                st.image(result_path, caption="Final Result (Transparent Background)", use_container_width=True)

                if draw_contour:
                    ruta_linia = extract_mountain_contour(result_path)
                    if ruta_linia:
                        st.success("Contorn extret correctament!")
                        st.image(ruta_linia, caption="Línia de la Muntanya", use_container_width=True)

                st.markdown("---")
                st.header("🗺️ Anàlisi Topogràfic i Alineament")

                foto_lat = 46.6358
                foto_lon = 12.3164
                foto_alt = 2405
                LA_TEVA_API_KEY = "8cfe74cb60bee3b3fff0d215fdb1bcda"

                # 1. Descarreguem el mapa
                dem_temporal = download_dynamic_dem(foto_lat, foto_lon, api_key=LA_TEVA_API_KEY, radi_km=35)

                # --- TOT EL MATCHING HA D'ANAR DINS D'AQUEST IF ---
                if dem_temporal:
                    # Generem la realitat (DEM)
                    perfil_virtual = get_360_profile(dem_temporal, foto_lat, foto_lon, foto_alt, max_dist_km=20)
                    st.pyplot(plot_360_profile(perfil_virtual))
                    os.remove(dem_temporal)

                    # Preparem la Foto
                    foto_array_1d = prepare_photo_profile(result_path)
                    
                    fig2, ax2 = plt.subplots(figsize=(8, 3))
                    ax2.plot(foto_array_1d, color='green', linewidth=2)
                    ax2.set_title("Perfil 1D extret de la FOTO")
                    st.pyplot(fig2)

                    # --- EXECUTEM ELS 3 ALGORISMES ---
                    angle_corr, fov_corr, sim_corr = find_best_match(foto_array_1d, perfil_virtual)
                    angle_pts, fov_pts, sim_pts = find_best_match_points(foto_array_1d, perfil_virtual)
                    angle_hits, fov_hits, score_hits = find_best_match_coincidence(foto_array_1d, perfil_virtual, tolerancia=0.05)

                    # --- MOSTREM ELS RESULTATS ---
                    st.subheader("🎯 Resultats de l'Alineament")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mètode Correlació", f"{angle_corr}º", f"{sim_corr*100:.1f}% certesa")
                    col2.metric("Mètode Punt a Punt", f"{angle_pts}º", f"{sim_pts:.1f}% certesa")
                    col3.metric("Mètode Coincidència", f"{angle_hits}º", f"{score_hits:.1f}% Hits")

                    if abs(angle_corr - angle_hits) <= 2: 
                        st.success("✅ Els mètodes coincideixen! Trobada confirmada.")
                    else:
                        st.warning("⚠️ Els algorismes dubten. Comprova els gràfics visuals.")

                    # --- DIBUIXEM LA SUPERPOSICIÓ FINAL (Fent servir el mètode de Coincidència) ---
                    fig_hits, ax_dem = plt.subplots(figsize=(12, 4))

                    ax_dem.plot(perfil_virtual, color='#1f77b4', linewidth=2, label="DEM 360º")
                    ax_dem.fill_between(range(360), perfil_virtual, color='#1f77b4', alpha=0.15)
                    ax_dem.set_xlabel("Azimut (Graus)")
                    ax_dem.set_ylabel("Angle d'Elevació DEM (Graus)", color='#1f77b4')
                    ax_dem.tick_params(axis='y', labelcolor='#1f77b4')

                    ax_dem.axvspan(angle_hits, angle_hits + fov_hits, color='green', alpha=0.15, label="Zona d'Encaix")

                    ax_foto = ax_dem.twinx()
                    foto_resized = cv2.resize(foto_array_1d.astype(np.float32).reshape(1, -1), (fov_hits, 1), interpolation=cv2.INTER_LINEAR)[0]
                    eix_x_foto = np.arange(angle_hits, angle_hits + fov_hits)

                    ax_foto.plot(eix_x_foto, foto_resized, color='green', linewidth=3, label="Perfil Foto")
                    ax_foto.set_ylabel("Píxels Foto", color='green')
                    ax_foto.tick_params(axis='y', labelcolor='green')
                    
                    ax_dem.set_xlim(0, max(359, angle_hits + fov_hits))
                    plt.title(f"Superposició Guanyadora (Azimut: {angle_hits}º | Zoom: {fov_hits}º)", fontweight='bold')
                    
                    st.pyplot(fig_hits)

                else:
                    st.error("No s'ha pogut connectar amb el mapa 3D.")
            else:
                st.error("Something went wrong. The resulting image was not saved properly.")