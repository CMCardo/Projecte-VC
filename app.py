import streamlit as st
import os
from photo_loader import sky_remove_cv2, sky_remove_ai, sky_remove_Laplace, sky_remove_hybrid
from functions import extract_mountain_contour
from DEM import get_360_profile, plot_360_profile, download_dynamic_dem
from Template_Matching import prepare_photo_profile, find_best_match

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

#checkbox for border
draw_contour = st.checkbox("Only mountain border (Siluet)")

# Process execution
if st.button("Process Image"):
    if not os.path.exists(image_name):
         st.error("Cannot process because the original image is missing.")
    else:
        with st.spinner("Processing image... Please wait."):
            
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
        
             

            if os.path.exists(result_path):
                st.image(result_path, caption="Final Result (Transparent Background)", use_container_width=True)

                #show border image
                if draw_contour:
                    ruta_linia = extract_mountain_contour(result_path)
                    if ruta_linia:
                        st.success("Contorn extret correctament!")
                        st.image(ruta_linia, caption="Línia de la Muntanya", use_container_width=True)

            else:
                st.error("Something went wrong. The resulting image was not saved properly.")


    print ("---------------------------------------------------------------")
    foto_lat = 46.6358
    foto_lon = 12.3164
    foto_alt = 2405
    #1, 12.311169072402626
    # ENGANXA AQUÍ LA TEVA CLAU D'OPENTOPOGRAPHY (entre les cometes)
    LA_TEVA_API_KEY = "8cfe74cb60bee3b3fff0d215fdb1bcda"

    # Li passem la clau a la funció
    dem_temporal = download_dynamic_dem(foto_lat, foto_lon, api_key=LA_TEVA_API_KEY, radi_km=35)

    if dem_temporal:
        perfil_virtual = get_360_profile(dem_temporal, foto_lat, foto_lon, foto_alt, max_dist_km=20)
        
        figura_perfil = plot_360_profile(perfil_virtual)
        st.pyplot(figura_perfil)
        
        os.remove(dem_temporal)
        print("🧹 Mapa temporal esborrat.")




    # 1. Preparem la línia verda de la foto i la invertim correctament
    foto_array_1d = prepare_photo_profile(result_path)

    # 2. Fem el Template Matching per buscar a quin punt del DEM s'assembla més
    angle, fov, similitud = find_best_match(foto_array_1d, perfil_virtual)

    # 3. Mostrem els resultats per pantalla!
    st.markdown("### 🎯 Resultats de l'Alineament")
    st.write(f"**Angle de visió (Azimut):** {angle}º")
    st.write(f"**Camp de visió (FOV / Zoom):** {fov}º")
    st.write(f"**Percentatge de certesa:** {similitud * 100:.2f}%")