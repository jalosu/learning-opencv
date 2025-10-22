"""
Paquete utils - Utilidades para procesamiento de imágenes y datos
"""

from .color_utils import (
    espacio_color,
    convertir_espacio_color,
    modificar_canal,
    _ensure_u8_3c,
    _mse
)

# También puedes importar funciones de otros módulos cuando los crees
# from .image_utils import redimensionar_imagen, recortar_imagen
# from .file_utils import cargar_imagen, guardar_imagen

__all__ = [
    'espacio_color',
    'convertir_espacio_color',
    'modificar_canal'
    # 'redimensionar_imagen',
    # 'cargar_imagen',
    # etc.
]

__version__ = '1.0.0'
