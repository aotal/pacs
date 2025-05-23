# utils.py
import os
import logging
import shutil
import math
import re # Para clean_filename_part
from typing import Any, Dict, List, Optional, Tuple, Union # Asegurar que Optional está aquí

# Importaciones de NumPy y Pandas solo si alguna función GENÉRICA aquí las necesita directamente.
# Si son para funciones que se han movido a otros módulos, no son necesarias aquí.
# import numpy as np # Necesario para las pruebas de serialización JSON si se mantienen los tipos np
# import pandas as pd

logger = logging.getLogger(__name__)

# --- Constantes y Diccionarios de Utilidad General ---

# Diccionario para traducciones (originado de tu decom.py)
TRANSLATION_DICT_200B7096: Dict[str, str] = {
    "Wall": "Paret",
    "Table": "Taula",
    # Añade más traducciones aquí según tus necesidades:
    # "ValorOriginalIngles": "TraduccionCatalan",
}

# --- Funciones de Utilidad ---

def get_translated_location(original_location_value: Union[str, bytes, None]) -> str:
    """
    Traduce el valor de localización usando TRANSLATION_DICT_200B7096.
    Devuelve el valor traducido, o el original si no se encuentra traducción,
    o un valor por defecto si la entrada es None o vacía.
    Maneja bytes decodificándolos.
    """
    default_translated_location = "UbicacioDesconeguda" 

    if original_location_value is None:
        logger.debug("Valor de localización original es None. Usando valor por defecto.")
        return default_translated_location
    
    if isinstance(original_location_value, bytes):
        try:
            original_location_value_str = original_location_value.decode('utf-8', errors='replace')
        except UnicodeDecodeError: 
            try:
                original_location_value_str = original_location_value.decode('latin-1', errors='replace')
                logger.debug("Valor de localización en bytes decodificado como latin-1.")
            except Exception: 
                logger.warning(f"No se pudo decodificar el valor de localización en bytes: {original_location_value!r}. Usando valor por defecto.")
                return default_translated_location
    else:
        original_location_value_str = str(original_location_value)

    cleaned_original_value = original_location_value_str.strip()

    if not cleaned_original_value:
        logger.debug("Valor de localización original vacío después de strip. Usando valor por defecto.")
        return default_translated_location
    
    translated = TRANSLATION_DICT_200B7096.get(cleaned_original_value, cleaned_original_value)
    
    if translated == cleaned_original_value and cleaned_original_value not in TRANSLATION_DICT_200B7096.values():
        logger.debug(f"No se encontró traducción para la localización '{cleaned_original_value}'. Se usará el valor original.")
    elif translated != cleaned_original_value:
        logger.debug(f"Localización '{cleaned_original_value}' traducida a '{translated}'.")
        
    return translated


def escribir_base64(ruta_archivo: str, cadena_base64: str) -> bool:
    """
    Escribe una cadena Base64 en un archivo de texto.
    Devuelve True si tiene éxito, False en caso contrario.
    """
    try:
        directorio = os.path.dirname(ruta_archivo)
        if directorio: 
             os.makedirs(directorio, exist_ok=True)
        with open(ruta_archivo, "w", encoding='utf-8') as f: 
            f.write(cadena_base64)
        logger.info(f"Cadena Base64 escrita en: {ruta_archivo}")
        return True
    except Exception as e:
        logger.exception(f"Error al escribir Base64 en {ruta_archivo}: {e}")
        return False


def obtener_ruta_salida(ruta_original: str, carpeta_destino: str, nueva_extension: str) -> str:
    """
    Genera una ruta de salida completa en una carpeta de destino con una nueva extensión,
    asegurando que el directorio de destino exista.
    """
    nombre_archivo_original = os.path.basename(ruta_original)
    nombre_base_original, _ = os.path.splitext(nombre_archivo_original)
    
    if not nueva_extension.startswith('.'):
        nueva_extension = '.' + nueva_extension
        
    nombre_archivo_destino = nombre_base_original + nueva_extension
    ruta_destino_completa = os.path.join(carpeta_destino, nombre_archivo_destino)
    
    directorio_final = os.path.dirname(ruta_destino_completa)
    if directorio_final: 
        os.makedirs(directorio_final, exist_ok=True)
    return ruta_destino_completa


def copiar_fichero(ruta_origen: str, ruta_destino: str) -> bool:
    """
    Copia un archivo de origen a destino, asegurando que el directorio de destino exista.
    Devuelve True si tiene éxito, False en caso contrario.
    """
    try:
        directorio_destino = os.path.dirname(ruta_destino)
        if directorio_destino:
            os.makedirs(directorio_destino, exist_ok=True)
        shutil.copy2(ruta_origen, ruta_destino) 
        logger.info(f"Archivo copiado de {ruta_origen} a {ruta_destino}")
        return True
    except FileNotFoundError:
        logger.error(f"Error al copiar: Archivo de origen no encontrado en {ruta_origen}")
        return False
    except Exception as e:
        logger.exception(f"Error al copiar archivo desde {ruta_origen} a {ruta_destino}: {e}")
        return False


def configurar_logging_aplicacion(log_file_path: Optional[str] = None, 
                                   level: int = logging.INFO,
                                   log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    """
    Configura el logging para la aplicación.
    Escribe en un archivo (opcional) y siempre en la consola.
    """
    handlers = [logging.StreamHandler()] 
    if log_file_path:
        try:
            log_dir = os.path.dirname(log_file_path)
            if log_dir: 
                os.makedirs(log_dir, exist_ok=True)
            handlers.append(logging.FileHandler(log_file_path, encoding='utf-8'))
            msg_log_file = log_file_path
        except OSError as e:
            print(f"ADVERTENCIA DE LOGGING: No se pudo crear el directorio para el log {log_file_path}: {e}. Logueando solo a consola.")
            msg_log_file = "No configurado (error al crear directorio)"
    else:
        msg_log_file = "No configurado"
            
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True # Para asegurar que se reconfigure si se llama varias veces (útil en algunos contextos de prueba)
    )
    logging.getLogger().info(f"Logging configurado. Nivel: {logging.getLevelName(level)}. "
                             f"Archivo de log: {msg_log_file}")


def clean_filename_part(part_value: Any, allowed_chars: str = "._-") -> str:
    """
    Limpia una cadena para ser usada como parte de un nombre de fichero,
    permitiendo solo caracteres alfanuméricos y los especificados en allowed_chars.
    Reemplaza los no permitidos por un guion bajo.
    """
    if part_value is None:
        return "Desconegut" 
    
    s_part_value = str(part_value)
    escaped_allowed_chars = re.escape(allowed_chars)
    pattern = r'[^a-zA-Z0-9' + escaped_allowed_chars + r']'
    
    cleaned_value = re.sub(pattern, '_', s_part_value)
    cleaned_value = re.sub(r'_+', '_', cleaned_value) 
    cleaned_value = cleaned_value.strip('_') 
    
    return cleaned_value if cleaned_value else "valor_net"


def get_file_extension(filepath: str) -> str:
    """Obtiene la extensión de un nombre de archivo (incluyendo el punto)."""
    if not filepath:
        return ""
    return os.path.splitext(os.path.basename(filepath))[1]


def convert_to_json_serializable(item: Any) -> Any:
    """
    Convierte tipos de datos (especialmente de NumPy y handling de NaN/inf)
    a tipos nativos de Python compatibles con la serialización JSON.
    """
    # Es necesario importar numpy aquí si las pruebas lo usan, o globalmente si es para el módulo
    import numpy as np 

    if isinstance(item, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        return [convert_to_json_serializable(elem) for elem in item]
    elif isinstance(item, np.ndarray):
        if np.issubdtype(item.dtype, np.floating):
            item_copy = np.copy(item) 
            item_copy[np.isinf(item_copy)] = np.nan 
            object_list = item_copy.astype(object).tolist()
            return [None if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else x for x in object_list]
        else: 
            return item.tolist()
    elif isinstance(item, np.bool_): return bool(item.item())
    elif isinstance(item, np.integer): return int(item.item())
    elif isinstance(item, np.floating): 
        scalar_item = item.item() 
        if math.isnan(scalar_item) or math.isinf(scalar_item):
            return None 
        return scalar_item
    elif isinstance(item, float):
        if math.isnan(item) or math.isinf(item):
            return None
        return item
    elif isinstance(item, (str, int, bool)) or item is None:
        return item
    else:
        logger.debug(f"Tipo no reconocido {type(item)} encontrado durante la serialización JSON. Convirtiendo a string.")
        return str(item)


if __name__ == '__main__':
    # Necesitamos numpy para las pruebas de serialización si usan tipos numpy
    import numpy as np 
    from pathlib import Path # Para manejo de rutas en pruebas

    configurar_logging_aplicacion(level=logging.DEBUG)

    logger.info("--- Pruebas para utils.py ---")
    
    logger.info("\n--- Pruebas de Traducción ---")
    print(f"'Wall' se traduce a: '{get_translated_location('Wall')}'")
    print(f"'Table ' se traduce a: '{get_translated_location('Table ')}'")
    print(f"'Floor' se traduce a: '{get_translated_location('Floor')}'")
    print(f"Tag en bytes (Wall): {get_translated_location(b'Wall')}")
    print(f"None se traduce a: '{get_translated_location(None)}'")
    print(f"Bytes no decodificables: {get_translated_location(b'\xff\xfe')}")


    logger.info("\n--- Pruebas de Rutas y Ficheros ---")
    test_output_dir = Path("output_utils_test")
    test_base64_dir = test_output_dir / "base64_files"
    test_copied_files_dir = test_output_dir / "copied_files"

    test_base64_dir.mkdir(parents=True, exist_ok=True)
    test_copied_files_dir.mkdir(parents=True, exist_ok=True)


    test_b64_path = obtener_ruta_salida("input/some_dicom.dcm", str(test_base64_dir), ".b64")
    print(f"Ruta de salida para Base64: {test_b64_path}")
    if escribir_base64(test_b64_path, "dGVzdGluZyBiYXNlNjQgd3JpdGluZw=="):
         if Path(test_b64_path).exists(): print(f"Fichero Base64 de prueba escrito en {test_b64_path}")

    test_copy_source = "test_source_file_utils.txt"
    test_copy_dest = str(test_copied_files_dir / "test_dest_file_utils.txt")
    with open(test_copy_source, "w") as f: f.write("Contenido de prueba para copiar.")
    if copiar_fichero(test_copy_source, test_copy_dest):
        if Path(test_copy_dest).exists(): print(f"Fichero de prueba copiado a {test_copy_dest}")

    logger.info("\n--- Pruebas de Limpieza de Nombres ---")
    print(f"Limpiando 'Detector/ID-01!': '{clean_filename_part('Detector/ID-01!')}'")
    # LÍNEAS CORREGIDAS:
    print(f"Limpiando 'KVP=70.5' (sin =): '{clean_filename_part('KVP=70.5', allowed_chars='._-')}'")
    print(f"Limpiando 'KVP=70.5' (con =): '{clean_filename_part('KVP=70.5', allowed_chars='._-=')}'")
    print(f"Limpiando '!@#$': '{clean_filename_part('!@#$')}'")
    print(f"Limpiando '__underscores__': '{clean_filename_part('__underscores__')}'")


    logger.info("\n--- Pruebas de Serialización JSON ---")
    data_to_serialize = {
        "np_array_float_nan_inf": np.array([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf]),
        "np_array_int": np.array([1,2,3]),
        "np_float32": np.float32(3.14159),
        "np_int64": np.int64(123),
        "py_float_nan": float('nan'),
        "py_float_inf": float('inf'),
        "py_list_mixed": [1, np.float32(np.nan), "text", {"nested_np_bool": np.bool_(True)}]
    }
    serializable_data = convert_to_json_serializable(data_to_serialize)
    import json
    try:
        json_string = json.dumps(serializable_data, indent=2)
        print(f"Datos serializados a JSON:\n{json_string}")
    except TypeError as te:
        print(f"Error al serializar a JSON: {te}")
        print(f"Datos que causaron el error: {serializable_data}")

    # Limpiar ficheros/directorios de prueba
    if Path(test_b64_path).exists(): Path(test_b64_path).unlink(missing_ok=True) # missing_ok para Python 3.8+
    if Path(test_copy_source).exists(): Path(test_copy_source).unlink(missing_ok=True)
    if Path(test_copy_dest).exists(): Path(test_copy_dest).unlink(missing_ok=True)
    if test_output_dir.exists(): shutil.rmtree(test_output_dir)