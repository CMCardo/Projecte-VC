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


def prepare_photo_profile(image_path):
    """
    Llegeix la imatge sense fons (RGBA) i n'extreu la silueta superior llegint la transparència.
    També aplica un filtre suau per eliminar soroll o punts solitaris.
    """
    # Llegim la imatge amb TOTS els seus canals (inclòs l'Alpha/Transparència si en té)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]
    
    perfil_1d = np.zeros(width)
    
    # Comprovem si la imatge té transparència (4 canals: Red, Green, Blue, Alpha)
    if img.shape[2] == 4:
        # Ens quedem només amb el canal de transparència (0 = Cel, 255 = Muntanya)
        mask = img[:, :, 3] 
    else:
        # Si no té transparència, la passem a grisos
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Escanegem columna a columna buscant el primer píxel sòlid
    for x in range(width):
        # Busquem el primer píxel que tingui un valor major a 10 (que no sigui transparent)
        y_muntanya = np.argmax(mask[:, x] > 10) 
        
        if y_muntanya == 0 and mask[0, x] <= 10:
            # Tota la columna és cel
            perfil_1d[x] = 0
        else:
            # Invertim l'eix: l'altura total menys la posició Y ens dona l'altura real
            perfil_1d[x] = height - y_muntanya
            
    # --- FILTRE DE NETEJA MAGIC ---
    # Apliquem un petit filtre "MedianBlur" a l'array matemàtic per eliminar 
    # punts flotants d'un sol píxel o talls estranys i deixar una línia suau i contínua.
    perfil_1d = cv2.GaussianBlur(perfil_1d.astype(np.float32).reshape(1, -1), (15, 1), 0)[0]
            
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




def extract_manual_profile(image_path):
    """
    Llegeix la imatge amb la línia vermella manual i en treu un perfil 1D.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Passem a format HSV per detectar el color VERMELL fàcilment
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # El vermell a OpenCV té dos rangs
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    height, width = red_mask.shape
    profile = np.full(width, np.nan) # Omplim amb "Not a Number" per defecte
    
 
    # Per a cada columna, busquem el primer píxel vermell (el més alt)
    for x in range(width):
        y_indices = np.where(red_mask[:, x] > 0)[0]
        if len(y_indices) > 0:
            # GIREM LA LÍNIA: Restem la posició Y a l'alçada total de la imatge
            profile[x] = height - y_indices[0]
            
    return profile

def compare_contours(profile_manual, profile_algo, tolerance_px=5):
    """
    Compara els dos perfils i retorna % d'encert, % d'error i la distància mitjana.
    """
    if profile_manual is None or profile_algo is None:
        return 0.0, 0.0, 0.0
        
    # Igualem longituds per si hi ha algun desajust
    min_len = min(len(profile_manual), len(profile_algo))
    p_man = profile_manual[:min_len]
    p_alg = profile_algo[:min_len]
    
    # Filtrem només les columnes on l'usuari ha dibuixat la línia vermella (ignorant NaNs)
    valid_cols = ~np.isnan(p_man) & ~np.isnan(p_alg)
    
    p_man_valid = p_man[valid_cols]
    p_alg_valid = p_alg[valid_cols]
    
    if len(p_man_valid) == 0:
        return 0.0, 0.0, 0.0
        
    # Calculem la distància absoluta en píxels per a cada columna
    diffs = np.abs(p_man_valid - p_alg_valid)
    
    # Comptem els píxels que estan dins del marge de tolerància
    matches = np.sum(diffs <= tolerance_px)
    errors = len(diffs) - matches
    
    match_pct = (matches / len(diffs)) * 100
    error_pct = (errors / len(diffs)) * 100
    mean_dist = np.mean(diffs)
    
    return match_pct, error_pct, mean_dist