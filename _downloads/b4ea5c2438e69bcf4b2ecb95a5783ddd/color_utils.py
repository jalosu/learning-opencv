import numpy as np
import cv2

def _ensure_u8_3c(img):
    """
    Garantiza que una imagen tenga 3 canales y esté en formato uint8.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Se esperaba imagen 3 canales, got shape={img.shape}")
    if img.dtype == np.uint8:
        return img
    mx = float(np.nanmax(img))
    if mx <= 1.0 + 1e-6:
        return (np.clip(img, 0, 1) * 255.0).round().astype(np.uint8)
    return np.clip(img, 0, 255).round().astype(np.uint8)

def _mse(a, b):
    """
    Calcula el Error Cuadrático Medio entre dos imágenes.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def espacio_color(img):
    """
    Posibles salidas:
      'GRAY', 'HSV', 'Lab/YCrCb', 'BGR/RGB', 'Desconocido'
    Devuelve dict con: espacio, dtype, shape, rangos por canal, y (opcional) errores de ciclo.
    """
    info = {"espacio":"Desconocido","dtype":None,"shape":None,"rangos":None,"ciclos":{}}
    if img is None: 
        return info

    info["dtype"] = str(img.dtype)
    info["shape"] = img.shape

    # 1) GRAY
    if img.ndim == 2:
        cmin, cmax = float(np.min(img)), float(np.max(img))
        info["espacio"] = "GRAY"
        info["rangos"]  = {"Canal": (round(cmin,2), round(cmax,2))}
        return info

    # 2) Estructura 3 canales
    if img.ndim != 3 or img.shape[2] != 3:
        return info

    # Rangos y estadísticas simples
    cmin = [float(np.min(img[...,i])) for i in range(3)]
    cmax = [float(np.max(img[...,i])) for i in range(3)]
    info["rangos"] = {f"C{i}": (round(cmin[i],2), round(cmax[i],2)) for i in range(3)}
    dtype = img.dtype

    # 3) HSV primero
    #  - uint8: H<=179 (regla rápida que suele funcionar)
    if dtype == np.uint8 and cmax[0] <= 180 and cmax[1] <= 255 and cmax[2] <= 255:
        info["espacio"] = "HSV"
        return info
    #  - float: H<=360; S,V<=1.1
    if np.issubdtype(dtype, np.floating):
        if cmax[0] <= 360 and cmax[1] <= 1.1 and cmax[2] <= 1.1:
            info["espacio"] = "HSV"
            return info

    # 4) Estrategia pragmática: Lab/YCrCb con umbrales MUY estrictos
    try:
        u8 = _ensure_u8_3c(img)
        
        # ciclo asumiendo YCrCb
        bgr_y  = cv2.cvtColor(u8, cv2.COLOR_YCrCb2BGR)
        rec_y  = cv2.cvtColor(bgr_y, cv2.COLOR_BGR2YCrCb)
        mse_y  = _mse(rec_y, u8)

        # ciclo asumiendo Lab
        bgr_l  = cv2.cvtColor(u8, cv2.COLOR_Lab2BGR)
        rec_l  = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2Lab)
        mse_l  = _mse(rec_l, u8)

        info["ciclos"] = {"YCrCb_mse": round(mse_y,3), "Lab_mse": round(mse_l,3)}

        # UMBRAL MUY ESTRICTO para Lab/YCrCb - SOLO si es casi perfecto
        if mse_y < 0.5 or mse_l < 0.5:  
            info["espacio"] = "Lab/YCrCb"
            return info
    except Exception:
        pass

    # 5) BGR/RGB como opción por defecto (caso más común)
    if dtype == np.uint8 and all(m <= 255 for m in cmax):
        info["espacio"] = "BGR/RGB"
        return info
    if np.issubdtype(dtype, np.floating) and all(m <= 1.0+1e-6 for m in cmax):
        info["espacio"] = "BGR/RGB"
        return info

    # 6) Nada cuadra
    return info

########################################################
def convertir_espacio_color(img, espacio_destino):
    """
    Convierte una imagen entre espacios de color de forma segura.
    """
    info_original = espacio_color(img)
    espacio_original = info_original["espacio"]
    
    # Mapeo de conversiones
    conversiones = {
        ("BGR/RGB", "HSV"): cv2.COLOR_BGR2HSV,
        ("BGR/RGB", "Lab"): cv2.COLOR_BGR2Lab,
        ("BGR/RGB", "GRAY"): cv2.COLOR_BGR2GRAY,
        ("HSV", "BGR/RGB"): cv2.COLOR_HSV2BGR,
        ("Lab", "BGR/RGB"): cv2.COLOR_Lab2BGR,
        ("GRAY", "BGR/RGB"): cv2.COLOR_GRAY2BGR,
    }
    
    clave = (espacio_original, espacio_destino)
    if clave in conversiones:
        return cv2.cvtColor(img, conversiones[clave])
    else:
        raise ValueError(f"Conversión no soportada: {espacio_original} -> {espacio_destino}")

###########################################################
def modificar_canal(canal, operacion='brillo', factor=1.5):
    """
    Modifica un canal según la operación especificada
    
    Args:
        canal: Canal a modificar
        operacion: 'brillo', 'contraste', 'invertir'
        factor: Factor de modificación
    """
    canal_mod = canal.copy().astype(np.float32)
    
    if operacion == 'brillo':
        # Aumentar brillo
        canal_mod = canal_mod * factor
        canal_mod = np.clip(canal_mod, 0, 255)
        
    elif operacion == 'contraste':
        # Aumentar contraste
        mean_val = np.mean(canal_mod)
        canal_mod = (canal_mod - mean_val) * factor + mean_val
        canal_mod = np.clip(canal_mod, 0, 255)
        
    elif operacion == 'invertir':
        # Invertir colores
        canal_mod = 255 - canal_mod
    
    return canal_mod.astype(np.uint8)
