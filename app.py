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

# Page configuration
st.set_page_config(page_title="Sky Remover App", layout="centered")

st.title("Mountain Sky Remover")
st.write("Select a photo number and the processing method to remove the sky.")

# Select photo number
photo_num = st.number_input(f"Select photo number (1-{MAX_NUM_PHOTOS}):", min_value=1, max_value=MAX_NUM_PHOTOS, value=1)

# Display the original image (Cerca dinàmica d'extensions)
image_name = None
extensions_suportades = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".webp", ".WEBP"]

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

# --- SECCIÓ DE COORDENADES (AUTOMÀTIQUES O MANUALS) ---
st.subheader("📍 Coordenades del punt d'observació")

# Intentem extreure el GPS de la foto abans de dibuixar les caixes
coords_auto = get_gps_from_photo(image_name) if os.path.exists(image_name) else None

if coords_auto:
    def_lat, def_lon, def_alt = coords_auto
    st.success("📸 GPS detectat automàticament a les metadades de la foto!")
else:
    # Valors per defecte (Tre Cime) si la foto no té GPS
    def_lat, def_lon, def_alt = 46.6358, 12.3164, 2405.0
    st.info("ℹ️ No s'ha trobat GPS a la foto. S'han carregat les coordenades per defecte.")

# Dibuixem les 3 caixes en columnes. L'usuari les pot modificar lliurement.
col_lat, col_lon, col_alt = st.columns(3)
foto_lat = col_lat.number_input("Latitud:", value=float(def_lat), format="%.6f")
foto_lon = col_lon.number_input("Longitud:", value=float(def_lon), format="%.6f")
foto_alt = col_alt.number_input("Altitud (metres):", value=float(def_alt), format="%.1f")

st.markdown("---")

# --- SECCIÓ DE LA FONT DEM I RADI ---
st.subheader("🗺️ Font del Mapa Topogràfic i Radi")
dem_options = {
    "NASA SRTMGL3 (Ideal per a zones fins a 60ºN, com els Alps o Dolomites)": "SRTMGL3",
    "Copernicus COP90 (Global, obligatori per al Nord d'Europa com Suècia)": "COP90"
}
selected_dem_label = st.selectbox("Selecciona quin model de satèl·lit vols fer servir:", list(dem_options.keys()))
selected_dem = dem_options[selected_dem_label]

# NOVA LÍNIA: Slider per triar els quilòmetres
radi_km_usuari = st.slider("Radi d'anàlisi i descàrrega del mapa (km):", min_value=10, max_value=80, value=35, step=5)

# NOVA LÍNIA AQUÍ: Col·loquem la tolerància a la configuració inicial
tolerancia_contorn = st.slider("Tolerància d'error del contorn (píxels):", min_value=1, max_value=50, value=20, step=1)

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

            # --- AVALUACIÓ DE PRECISIÓ DE L'ALGORISME ---
                ruta_manual = f"border_test_photo/IMG_{photo_num}.jpg" # Assegura't que el format i nom coincideixin
                
                if os.path.exists(ruta_manual):
                    st.markdown("---")
                    st.subheader("⚖️ Avaluació de Precisió del Contorn")
                    
                    with st.spinner("Comparant línia de l'algorisme amb el Ground Truth manual..."):
                        # Extraiem el perfil de la línia vermella manual
                        perfil_manual = extract_manual_profile(ruta_manual)
                        
                        # Extraiem el perfil que ha generat l'algorisme (agafant la imatge transparent)
                        perfil_algo = prepare_photo_profile(result_path)
                        
                        # CANVI AQUÍ: Eliminem la línia 'tolerancia = 20' 
                        # i usem la variable 'tolerancia_contorn' que ve del slider de dalt
                        encert_pct, error_pct, dist_mitjana = compare_contours(perfil_manual, perfil_algo, tolerance_px=tolerancia_contorn)
                        
                        # Creem les columnes (només una vegada)
                        col1, col2, col3 = st.columns(3)
                        
                        # Mostrem les mètriques fent servir la variable del lliscador de dalt
                        col1.metric(f"Coincidència (±{tolerancia_contorn}px)", f"{encert_pct:.1f} %")
                        col2.metric("Error (Fora de línia)", f"{error_pct:.1f} %")
                        col3.metric("Desviació mitjana", f"{dist_mitjana:.1f} píxels")
                        # --- NOU BLOC: VISUALITZACIÓ DE LA COMPARATIVA ---
                        st.markdown("#### 🔍 Visualització de la Comparativa")
                        fig_comp, ax_comp = plt.subplots(figsize=(10, 3.5))
                        
                        # Dibuixem les dues línies per veure-les superposades
                        ax_comp.plot(perfil_algo, label="Línia de l'Algorisme (Verda)", color='green', linewidth=2)
                        ax_comp.plot(perfil_manual, label="El teu Dibuix (Vermell)", color='red', linewidth=2, linestyle='dashed')
                        
                        ax_comp.set_title("Superposició dels dos contorns (Ground Truth vs Algorisme)")
                        ax_comp.set_xlabel("Píxels (Amplada de la foto)")
                        ax_comp.set_ylabel("Posició (Alçada)")
                        ax_comp.legend()
                        st.pyplot(fig_comp)
                        # -------------------------------------------------
                        
                        if encert_pct > 90:
                            st.success("L'algorisme ha fet un retall pràcticament perfecte respecte al teu dibuix manual.")
                        elif encert_pct > 70:
                            st.info("L'algorisme és bo, però s'ha desviat en algunes zones complexes o ombres.")
                        else:
                            st.error("L'algorisme ha fallat força. Intenta fer servir un altre mètode (AI o Laplace).")
                else:
                    st.info(f"💡 No s'ha trobat cap dibuix manual a `{ruta_manual}` per avaluar la precisió d'aquesta foto.")
                
                st.markdown("---")
                st.header("🗺️ Anàlisi Topogràfic i Alineament")

                LA_TEVA_API_KEY = "8cfe74cb60bee3b3fff0d215fdb1bcda"

                # 1. Descarreguem el mapa fent servir la variable del lliscador (radi_km_usuari)
                dem_temporal = download_dynamic_dem(foto_lat, foto_lon, api_key=LA_TEVA_API_KEY, radi_km=radi_km_usuari, dem_type=selected_dem)

                # --- TOT EL MATCHING HA D'ANAR DINS D'AQUEST IF ---
                if dem_temporal:
                    # Generem la realitat (DEM) usant el lliscador
                    perfil_virtual = get_360_profile(dem_temporal, foto_lat, foto_lon, foto_alt, max_dist_km=radi_km_usuari)
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

                    # --- DIBUIXEM ELS GRÀFICS PER SEPARAT ---
                    st.markdown("---")
                    st.header("📈 Visualització de l'Encaix per Mètode")

                    def plot_basic_match(angle, fov, titol, color_zona):
                        fig, ax_dem = plt.subplots(figsize=(10, 3.5))
                        
                        ax_dem.plot(perfil_virtual, color='#1f77b4', linewidth=2)
                        ax_dem.fill_between(range(360), perfil_virtual, color='#1f77b4', alpha=0.15)
                        ax_dem.set_xlim(0, max(359, angle + fov))
                        ax_dem.set_xlabel("Azimut (Graus)")
                        ax_dem.set_ylabel("Elevació DEM", color='#1f77b4')
                        ax_dem.axvspan(angle, angle + fov, color=color_zona, alpha=0.2, label="Zona d'Encaix")
                        
                        ax_foto = ax_dem.twinx()
                        foto_pixels_resized = cv2.resize(foto_array_1d.astype(np.float32).reshape(1, -1), (fov, 1), interpolation=cv2.INTER_LINEAR)[0]
                        eix_x = np.arange(angle, angle + fov)
                        
                        ax_foto.plot(eix_x, foto_pixels_resized, color='green', linewidth=2, alpha=0.5, label="Perfil Foto")
                        ax_foto.set_ylabel("Píxels Foto", color='green')
                        
                        plt.title(titol, fontweight='bold')
                        return fig, ax_foto, foto_pixels_resized, eix_x

                    # 1. Gràfic de Correlació
                    fig_corr, _, _, _ = plot_basic_match(angle_corr, fov_corr, f"1. Correlació (Azimut: {angle_corr}º | Zoom: {fov_corr}º)", 'orange')
                    st.pyplot(fig_corr)

                    # 2. Gràfic Punt a Punt
                    fig_pts, _, _, _ = plot_basic_match(angle_pts, fov_pts, f"2. Distància Punt a Punt (Azimut: {angle_pts}º | Zoom: {fov_pts}º)", 'purple')
                    st.pyplot(fig_pts)

                    # 3. Gràfic de Coincidència (AMB ELS PUNTS DE COLORS)
                    fig_hits, ax_foto, foto_pixels_resized, eix_x = plot_basic_match(angle_hits, fov_hits, f"3. Coincidència de Punts (Azimut: {angle_hits}º | Hits: {score_hits:.1f}%)", 'green')
                    
                    dem_extended = np.concatenate((perfil_virtual, perfil_virtual))
                    dem_window = dem_extended[angle_hits : angle_hits + fov_hits]
                    
                    foto_norm = (foto_array_1d - np.min(foto_array_1d)) / (np.max(foto_array_1d) - np.min(foto_array_1d) + 1e-5)
                    foto_res_norm = cv2.resize(foto_norm.reshape(1, -1), (fov_hits, 1), interpolation=cv2.INTER_LINEAR)[0]
                    dem_window_norm = (dem_window - np.min(dem_window)) / (np.max(dem_window) - np.min(dem_window) + 1e-5)
                    
                    tolerancia = 0.05
                    mask_hits = np.abs(foto_res_norm - dem_window_norm) < tolerancia
                    
                    ax_foto.scatter(eix_x[mask_hits], foto_pixels_resized[mask_hits], color='lime', s=25, zorder=5, label='Coincidència (SÍ)')
                    ax_foto.scatter(eix_x[~mask_hits], foto_pixels_resized[~mask_hits], color='red', s=25, zorder=5, label='Error (NO)')
                    ax_foto.legend(loc='upper right')
                    
                    st.pyplot(fig_hits)

                    # --- IDENTIFICACIÓ DE CIMS ---
                    st.markdown("---")
                    st.header("🏔️ Identificació del Cim Principal")
                    
                    with st.spinner("Creuant coordenades amb la base de dades d'OpenStreetMap..."):
                        
                        # Rebem les 3 variables de la funció, ara usant radi_km_usuari
                        cims_visibles, total_zona, tots_els_cims = get_visible_peaks(foto_lat, foto_lon, angle_hits, fov_hits, radius_km=radi_km_usuari)
                        
                        st.info(f"📡 Radar: S'han detectat **{total_zona}** cims en un radi de {radi_km_usuari}km al teu voltant.")
                        
                        if cims_visibles:
                            # 1. EL CIM MÉS PROBABLE (El primer de la llista, que és el més alt)
                            cim_estrella = cims_visibles[0]
                            
                            st.success("✅ Encaix completat! S'ha trobat el subjecte principal de la foto.")
                            
                            st.markdown(f"### 👑 **{cim_estrella['Nom del Cim']}**")
                            colA, colB = st.columns(2)
                            colA.metric("Altitud", f"{cim_estrella['Altitud (m)']} metres")
                            colB.metric("Azimut a la foto", f"{cim_estrella['Azimut (º)']}º")
                            
                            st.write("Aquest és el cim més rellevant identificat just a la direcció on apunta la teva foto.")
                            
                            # 2. ELS ALTRES CIMS VISIBLES (Amagats en un desplegable)
                            if len(cims_visibles) > 1:
                                with st.expander(f"👀 Veure els altres {len(cims_visibles) - 1} cims secundaris dins l'enquadrament"):
                                    # Fem un dataframe excloent-ne el primer (que ja està destacat a dalt)
                                    df_secundaris = pd.DataFrame(cims_visibles[1:])
                                    st.dataframe(df_secundaris[['Nom del Cim', 'Azimut (º)', 'Altitud (m)']], use_container_width=True)
                                    
                        else:
                            st.warning("⚠️ L'algorisme està apuntant cap a una zona on no hi ha cims visibles (revisa que la línia verda de la foto no tingui defectes).")
                            
                        # --- 3. MODE DEBUG: TOTS ELS CIMS (Amagat per defecte) ---
                        if tots_els_cims:
                            with st.expander("🔍 Mode Desenvolupador: Veure el radar complet (Tots els cims de la zona)"):
                                st.write("Aquesta taula et mostra absolutament tot el que tens al voltant. Si el teu cim surt marcat amb una ❌, vol dir que l'algorisme l'ha deixat fora de la caixa verda.")
                                st.dataframe(pd.DataFrame(tots_els_cims), use_container_width=True)

                else:
                    st.error("No s'ha pogut connectar amb el mapa 3D.")
            else:
                st.error("Something went wrong. The resulting image was not saved properly.")