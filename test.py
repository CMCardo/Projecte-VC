from photo_loader import sky_remove_cv2, sky_remove_ai, sky_remove_Laplace, sky_remove_hybrid



def test_photos_ia():


    models_disponibles = {
        "1": "u2net",   #Good for general objects, but might struggle with complex landscapes
        "2": "isnet-general-use", #Best for landscapes and nature. Highly accurate for complex scenes and textures
        "3": "u2netp", #Lightweight version of the default model. Faster, but slightly less accurate on edges
        "4": "silueta" #Extremely small and fast model. Great for quick tests but sacrifices precision
    }
        
    model_num = input("What model of AI do you want: 1. u2net (default)   2. isnet-general-use   3.u2netp   4. silueta: ")
    model = models_disponibles.get(model_num, "u2net")

    for num in range(1,44): 
        sky_remove_ai(num, model)




def test_photo_laplace():

    tolerance = 3

    for num in range(1,44): 
        sky_remove_Laplace(num, tolerance)


def test_photo_hybrid():
    tolerance = 3

    for num in range(1,44): 
        sky_remove_hybrid(num, tolerance)

