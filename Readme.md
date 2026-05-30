# 🏔️ Mountain Sky Remover & Peak Identifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

**University Project - Computer Vision**

The objective of this project is to detect and identify mountain peaks from photographs using almost exclusively Computer Vision and 1D signal processing, bypassing the need for heavy 3D rendering engines or unreliable smartphone compass sensors.

## ⚙️ How it Works

1. **GPS Telemetry:** Extracts Latitude, Longitude, and Altitude directly from the EXIF metadata of the handmade photos.
2. **Sky Removal & Silhouette Extraction:** Processes the image to remove the sky and extract a 1D curve of the real horizon. It features 4 interchangeable methods:
   - Color Thresholding (HSV)
   - Texture filtering (Laplace)
   - Artificial Intelligence (U^2-Net / Rembg)
   - Hybrid approach
3. **DEM Ray-Casting:** Uses the given GPS position to download a Digital Elevation Model (Copernicus COP90 or NASA SRTMGL3) via OpenTopography. It calculates a 360º panoramic 1D profile using ray-casting.
4. **Multi-scale Template Matching:** The system slides the photo's 1D curve over the DEM's 360º curve. By dynamically resizing the image to test different Fields of View (FOV), it finds the exact Camera Azimuth and scale using techniques like Normalized Cross-Correlation (NCC) and Coincidence Hits.
5. **Peak Labeling:** Queries the **OpenStreetMap (Overpass API)** to identify and label the visible peaks within the exact calculated field of view.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name




   Install the dependencies:
Make sure you have Python installed, then run:

pip install -r requirements.txt



Environment Variables:
To download the DEM maps, you need an API Key from OpenTopography.
Create a .env file in the root directory and add your key:
Fragmento de código

OPENTOPOGRAPHY_API_KEY=your_api_key_here



To launch the web interface, run the following command in your terminal:
Bash

python -m streamlit run app.py
