import numpy as np
import cv2

def find_best_match(foto_profile, dem_profile, min_fov=30, max_fov=100):
    """
    Compara el perfil 1D de la foto amb el perfil 360º del DEM buscant el millor encaix.
    
    :param foto_profile: Array 1D amb les altures de la muntanya a la foto.
    :param dem_profile: Array 1D de 360 valors amb l'horitzó del DEM.
    :param min_fov: Graus mínims que creiem que pot ocupar la foto (Zoom).
    :param max_fov: Graus màxims que pot ocupar la foto (Gran Angular).
    :return: (millor_angle, millor_fov, percentatge_similitud)
    """
    
    best_corr = -1.0
    best_fov = 0
    best_angle = 0
    
    # 1. TRUC DEL WRAP-AROUND (Cilindre 360º)
    # Dupliquem el DEM i l'enganxem darrere seu (720 graus en total).
    # Així, si la foto comença al grau 350 i acaba al 20, no donarà error per sortir de l'array.
    dem_extended = np.concatenate((dem_profile, dem_profile))
    
    # OpenCV necessita arrays 2D per fer el matchTemplate, així que li afegim una dimensió (1 fila, N columnes)
    dem_2d = dem_extended.reshape(1, -1).astype(np.float32)
    
    # Assegurem que la foto està en format correcte abans del bucle
    foto_1d = np.array(foto_profile, dtype=np.float32)
    
    # 2. BUCLE MULTI-ESCALA (Buscant el FOV perfecte)
    for fov in range(min_fov, max_fov + 1):
        
        # Redimensionem la línia de la foto perquè mesuri exactament 'fov' píxels/graus.
        # cv2.resize espera una imatge, per això passem l'array a forma (1, longitud)
        foto_resized_2d = cv2.resize(foto_1d.reshape(1, -1), (fov, 1), interpolation=cv2.INTER_LINEAR)
        
        # 3. CORRELACIÓ (Encaix geomètric)
        # TM_CCOEFF_NORMED ignora si la foto està en "píxels" i el DEM en "graus" d'alçada.
        # Només es fixa en si la FORMA (les pujades i baixades) coincideix.
        result = cv2.matchTemplate(dem_2d, foto_resized_2d, cv2.TM_CCOEFF_NORMED)
        
        # Busquem el valor més alt d'aquesta passada
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Si aquest FOV encaixa millor que l'anterior, guardem el rècord
        if max_val > best_corr:
            best_corr = max_val
            best_fov = fov
            best_angle = max_loc[0] # max_loc retorna (X, Y), ens quedem la X (l'angle)
            
    # Assegurem que l'angle final estigui entre 0 i 359 (pel wrap-around)
    best_angle = best_angle % 360
            
    return best_angle, best_fov, best_corr


def prepare_photo_profile(binary_mask_path):
    """
    Llegeix la màscara final sense cel i n'extreu un array 1D d'altures.
    """
    # Llegim la imatge en escala de grisos (el cel serà negre o transparent, la muntanya blanca/gris)
    img = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    
    perfil_1d = np.zeros(width)
    
    # Per cada columna de la foto, busquem on comença la muntanya
    for x in range(width):
        # Busquem el primer píxel que no sigui cel (baixant des de dalt)
        y_muntanya = np.argmax(img[:, x] > 0) 
        
        if y_muntanya == 0 and img[0, x] == 0:
            # Si no hi ha muntanya en aquesta columna (tot és cel)
            perfil_1d[x] = 0
        else:
            # INVERTIM L'EIX: Restem l'alçada total perquè els pics tinguin valors positius grans
            perfil_1d[x] = height - y_muntanya
            
    return perfil_1d




def find_best_match_points(foto_profile, dem_profile, min_fov=30, max_fov=100):

    best_error = float('inf') # Comencem amb un error infinitament gran
    best_fov = 0
    best_angle = 0
    
    # 1. Preparem el DEM (truc del wrap-around)
    dem_extended = np.concatenate((dem_profile, dem_profile))
    
    # 2. Normalitzem la foto tota sencera de 0 a 1
    # Evitem dividir per zero sumant un número minúscul (1e-5)
    foto_norm = (foto_profile - np.min(foto_profile)) / (np.max(foto_profile) - np.min(foto_profile) + 1e-5)
    
    # 3. Bucle Multi-escala
    for fov in range(min_fov, max_fov + 1):
        
        # Redimensionem la foto normalitzada
        foto_resized = cv2.resize(foto_norm.reshape(1, -1), (fov, 1), interpolation=cv2.INTER_LINEAR)[0]
        
        # 4. Fem lliscar aquesta foto per cada angle del DEM
        for angle in range(360):
            # Retallem el tros de DEM que toca analitzar ara
            dem_window = dem_extended[angle : angle + fov]
            
            # Normalitzem AQUEST tros de DEM de 0 a 1
            dem_window_norm = (dem_window - np.min(dem_window)) / (np.max(dem_window) - np.min(dem_window) + 1e-5)
            
            # 5. MATEMÀTIQUES PURES: Càlcul d'error punt a punt
            # Restem el valor del DEM al de la Foto, treiem els negatius (abs) i fem la mitjana
            error_actual = np.mean(np.abs(foto_resized - dem_window_norm))
            
            # Si aquest error és més petit que el rècord anterior, el guardem!
            if error_actual < best_error:
                best_error = error_actual
                best_angle = angle
                best_fov = fov
                
    best_angle = best_angle % 360
    
    # Convertim l'error (de 0 a 1) en un percentatge d'encert perquè sigui fàcil de llegir
    percentatge_encert = (1.0 - best_error) * 100
                
    return best_angle, best_fov, percentatge_encert



def find_best_match_coincidence(foto_profile, dem_profile, min_fov=30, max_fov=100, tolerancia=0.05):
    """
    Fa lliscar la línia de la foto sobre el DEM i compta quants punts coincideixen
    (estan pràcticament l'un a sobre de l'altre dins d'un marge de tolerància).
    Retorna l'angle i el FOV amb el percentatge més alt de punts "Hit".
    """
    best_score = -1.0 # Busquem el percentatge d'encerts més alt
    best_fov = 0
    best_angle = 0
    
    # 1. Preparem el DEM (truc del cilindre de 360º)
    dem_extended = np.concatenate((dem_profile, dem_profile))
    
    # 2. Normalitzem la foto de 0 a 1
    foto_norm = (foto_profile - np.min(foto_profile)) / (np.max(foto_profile) - np.min(foto_profile) + 1e-5)
    
    for fov in range(min_fov, max_fov + 1):
        
        # Redimensionem la foto perquè tingui exactament 'fov' punts
        foto_resized = cv2.resize(foto_norm.reshape(1, -1), (fov, 1), interpolation=cv2.INTER_LINEAR)[0]
        
        for angle in range(360):
            # Retallem el tros de DEM que toca ara
            dem_window = dem_extended[angle : angle + fov]
            
            # Normalitzem aquest tros de DEM de 0 a 1
            dem_window_norm = (dem_window - np.min(dem_window)) / (np.max(dem_window) - np.min(dem_window) + 1e-5)
            
            # 3. EL TEU MÈTODE: Càlcul de coincidències positives (Hits)
            # Calculem la distància exacta entre cada parella de punts
            diferencies = np.abs(foto_resized - dem_window_norm)
            
            # Comptem QUANTS punts tenen una distància més petita que la nostra tolerància (5%)
            punts_coincidents = np.sum(diferencies < tolerancia)
            
            # Calculem quin percentatge de la línia ha coincidit
            score_actual = (punts_coincidents / fov) * 100
            
            # Si aquesta posició té més coincidències que el nostre rècord, la guardem!
            if score_actual > best_score:
                best_score = score_actual
                best_angle = angle
                best_fov = fov
                
    best_angle = best_angle % 360
                
    return best_angle, best_fov, best_score