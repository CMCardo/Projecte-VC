VC-Proj/
├── Cache/
├── Photos/
├── .env
├── .gitignore
└── src/
    ├── main.py                 # (Punt d'entrada. Només contindrà el codi per llançar Streamlit)
    ├── app.py                  # (Interfície d'usuari i orquestració visual)
    │
    ├── segmentation.py         # 1. RETALL DE LA FOTO
    │   # Funcions: sky_remove_cv2, sky_remove_ai, sky_remove_laplace, sky_remove_hybrid
    │   # Canvi clau: Ara retornaran l'array de la imatge, NO guardaran res al disc.
    │
    ├── profiling.py            # 2. EXTRACCIÓ DE SILUETES
    │   # Funcions: prepare_photo_profile (crea vector 1D) i extract_mountain_contour (dibuixa línia verda)
    │
    ├── dem_engine.py           # 3. GESTIÓ DEL MAPA 3D
    │   # Funcions: download_dynamic_dem (amb el nou sistema robust de memòria cau) i get_360_profile (ray-casting)
    │
    ├── matching.py             # 4. ENCAIX DELS PERFILS
    │   # Funcions: find_best_match (Correlació), find_best_match_points i find_best_match_coincidence
    │
    ├── peaks_api.py            # 5. IDENTIFICACIÓ DE PICS
    │   # Funcions: get_visible_peaks i calculate_bearing (connexió amb OpenStreetMap)
    │
    ├── evaluation.py           # 6. GROUND TRUTH I MÈTRIQUES
    │   # Funcions: extract_manual_profile i compare_contours
    │
    └── utils.py                # 7. EINES GENERALS
        # Funcions: get_gps_from_photo (Llegir metadades EXIF)