from photo_loader import sky_remove_cv2, sky_remove_ai, sky_remove_Laplace, sky_remove_hybrid
import os
import cv2
import numpy as np

def ask_photo_num_and_model(max_num_photo):


    #demanar imatge
    while True:
        entrada = input(f"What image do you want to use (1-{max_num_photo}): ")
        if entrada.isdigit():
            num = int(entrada)
            if 1 <= num <= max_num_photo:
                break
        print(f"Invalid number. What image do you want to use (1-{max_num_photo}): ")


    methot = input("What methot do you want to use: 1. cv2   2. AI (default)    3. Laplace:    4, Hybrid")

    if methot == "1":

        sky_remove_cv2(num)

    if methot == "2":

        models_disponibles = {
            "1": "u2net",   #Good for general objects, but might struggle with complex landscapes
            "2": "isnet-general-use", #Best for landscapes and nature. Highly accurate for complex scenes and textures
            "3": "u2netp", #Lightweight version of the default model. Faster, but slightly less accurate on edges
            "4": "silueta" #Extremely small and fast model. Great for quick tests but sacrifices precision
        }
        
        model_num = input("What model of AI do you want: 1. u2net (default)   2. isnet-general-use   3.u2netp   4. silueta: ")
        model = models_disponibles.get(model_num, "u2net")

        sky_remove_ai(num, model)

    if methot == "3":

        tolerance = input("What tolerance do you want to use, 3 recommended")
        sky_remove_Laplace(num, tolerance)

    else:
        tolerance = input("What tolerance do you want to use, 3 recommended")
        sky_remove_hybrid(num, tolerance)



    

def extract_mountain_contour(result_path):

    if not os.path.exists(result_path):
        print(f"Error: Image not found: {result_path}")
        return None

    #load the 4 channels
    image = cv2.imread(result_path, cv2.IMREAD_UNCHANGED)

    # Check if there's 4 channels (rgba)
    if image.shape[2] != 4:
        print("Error: Image has no transparency")
        return None

    #mountaun alfa = white | sky alfa = black
    alpha_channel = image[:, :, 3]

    pad = 10
    #increase image size to delete the border created at the image limits
    padded_alpha = cv2.copyMakeBorder(alpha_channel, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    #search for borders
    border, _ = cv2.findContours(padded_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # black space same size as the original image
    h_pad, w_pad = padded_alpha.shape
    padded_canvas = np.zeros((h_pad, w_pad, 3), dtype=np.uint8)

    # Draw border line (green color)
    cv2.drawContours(padded_canvas, border, -1, (0, 255, 0), 3)

    # Cut the image to original size
    imatge_linia = padded_canvas[pad:-pad, pad:-pad]

    output_folder = "Contours"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    nom_arxiu = os.path.basename(result_path)
    rute_border = f"{output_folder}/line_{nom_arxiu}"
    
    cv2.imwrite(rute_border, imatge_linia)
    
    return rute_border