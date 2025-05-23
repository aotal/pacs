# baml_classification.py
import asyncio
import base64
import io
import logging
import os
import time # Necesario para algunos callbacks de tenacity si se usan
from typing import Optional, Any

from PIL import Image as PilImage
import numpy as np
from dotenv import load_dotenv

# Tenacity imports
from tenacity import (
    retry, 
    stop_after_attempt, # Reintentar un número específico de veces
    wait_exponential,   # Espera exponencial entre reintentos
    retry_if_exception_type, # Reintentar solo para tipos específicos de excepción
    before_sleep_log    # Loguear antes de dormir para el reintento
)

try:
    from baml_client import b, reset_baml_env_vars
    from baml_py import Image as BamlImage 
    # Asumiendo que BamlClientHttpError está disponible para ser capturada
    from baml_py.internal_monkeypatch import BamlClientHttpError # Ajusta esta importación si es diferente
except ImportError as e:
    logging.critical(f"No se pudieron importar los módulos de BAML o BamlClientHttpError. Error: {e}")
    b = None 
    BamlImage = None
    BamlClientHttpError = Exception # Fallback genérico si no se puede importar el error específico
except Exception as e_baml_misc: # Captura otros errores de importación de BAML
    logging.critical(f"Error inesperado importando módulos de BAML: {e_baml_misc}")
    b = None
    BamlImage = None
    BamlClientHttpError = Exception


logger = logging.getLogger(__name__)

# --- Configuración Inicial de BAML (como estaba) ---
try:
    if load_dotenv(): logger.info("Variables de entorno (.env) cargadas para BAML.")
    else: logger.warning("No se encontró .env. BAML podría no configurarse si depende de él.")
    if b:
        reset_baml_env_vars(dict(os.environ))
        logger.info("Variables de entorno BAML reseteadas/aplicadas al cliente.")
    elif BamlImage is None: logger.error("Cliente BAML ('b') o 'BamlImage' no importados. Clasificación BAML no funcionará.")
    else: logger.error("Cliente BAML ('b') no importado. Clasificación BAML no funcionará.")
except Exception as e_baml_init:
    logger.exception(f"Error inicializando BAML: {e_baml_init}")


def _convert_pixel_array_to_png_base64(pixel_array_visual: np.ndarray) -> Optional[str]:
    # ... (esta función permanece igual que en la última versión funcional) ...
    if not isinstance(pixel_array_visual, np.ndarray):
        logger.error("Entrada para conversión a base64 no es array numpy.")
        return None
    if pixel_array_visual.ndim != 2:
        logger.error(f"Array de píxeles debe ser 2D, pero tiene {pixel_array_visual.ndim} dims.")
        return None
    try:
        img_array_8bit: np.ndarray
        if pixel_array_visual.dtype == np.uint8:
            img_array_8bit = pixel_array_visual
        else:
            min_val, max_val = np.min(pixel_array_visual), np.max(pixel_array_visual)
            if max_val == min_val:
                img_array_8bit = np.zeros_like(pixel_array_visual, dtype=np.uint8)
            else:
                img_array_8bit = np.interp(pixel_array_visual, (min_val, max_val), (0, 255)).astype(np.uint8)
            logger.debug(f"Pixel array ({pixel_array_visual.dtype}) normalizado a 8-bit desde rango [{min_val}, {max_val}].")
        
        pil_img = PilImage.fromarray(img_array_8bit)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        base64_str = base64.b64encode(png_bytes).decode('utf-8')
        logger.info("Pixel array convertido a PNG base64 exitosamente.")
        return base64_str
    except Exception as e:
        logger.error(f"Error convirtiendo pixel_array a PNG base64: {e}", exc_info=True)
        return None

# --- Función de Clasificación con Reintentos usando Tenacity ---
def _is_rate_limit_error(exception: BaseException) -> bool:
    """Comprueba si la excepción es un error de límite de tasa de BAML (HTTP 429)."""
    return isinstance(exception, BamlClientHttpError) and exception.status_code == 429

def _get_retry_delay_from_baml_error(exception: BamlClientHttpError) -> Optional[float]:
    """Intenta extraer el 'retryDelay' del mensaje de error de Gemini."""
    try:
        import json
        # El mensaje de error de Gemini tiene un JSON incrustado después de "Too Many Requests. "
        # Esto es frágil y depende del formato exacto del mensaje.
        json_part_str = exception.message.split("Too Many Requests. ", 1)[1]
        error_details = json.loads(json_part_str)
        for detail in error_details.get("error", {}).get("details", []):
            if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo" and "retryDelay" in detail:
                delay_str = detail["retryDelay"].replace("s", "")
                return float(delay_str)
    except (IndexError, json.JSONDecodeError, ValueError, TypeError) as e_parse:
        logger.warning(f"No se pudo parsear retryDelay del error 429 de BAML, se usará el backoff exponencial. Error de parseo: {e_parse}")
    return None

# Configuración del decorador de reintentos de Tenacity
@retry(
    stop=stop_after_attempt(4),  # Reintentar hasta 3 veces más (total 4 intentos)
    wait=wait_exponential(multiplier=2, min=2, max=60), # Espera exponencial: 2s, 4s, 8s, etc., hasta 60s
    retry=retry_if_exception_type(BamlClientHttpError), # Reintentar solo para errores HTTP de BAML
    before_sleep=before_sleep_log(logger, logging.INFO) # Loguear antes de cada espera
)
def _clasificar_con_baml_tenacity(imagen_base64: str) -> str:
    """
    Clasifica una imagen usando BAML, con reintentos automáticos para errores HTTP recuperables.
    Mapea el resultado a FDT, MTF, BC, o DESCONOCIDA.
    """
    # Los checks de if not imagen_base64 y if b is None se mueven a la función llamadora
    # para que tenacity no reintente por esos errores.
    
    logger.info("Intentando clasificación BAML (con tenacity)...")
    baml_input_image = BamlImage.from_base64("image/png", imagen_base64)
    result = b.ClassifyImage(img=baml_input_image) # Esta es la llamada que puede fallar
    
    original_baml_classification = ""
    if hasattr(result, 'value'):
        original_baml_classification = str(result.value).strip()
        logger.info(f"Clasificación BAML recibida (desde .value): {original_baml_classification}")
    elif isinstance(result, str):
        original_baml_classification = result.strip()
        logger.info(f"Clasificación BAML recibida (string directo): {original_baml_classification}")
    else:
        original_baml_classification = str(result).strip() if result is not None else ""
        logger.warning(f"Resultado BAML ({type(result)}) sin '.value' y no es string. Convertido a: '{original_baml_classification}'")

    if original_baml_classification == "Type1": mapped_classification = "FDT"
    elif original_baml_classification == "Type2": mapped_classification = "MTF"
    elif original_baml_classification == "Type3": mapped_classification = "TOR" # Corregido
    elif not original_baml_classification:
        logger.warning("Clasificación BAML original vacía. Mapeando a DESCONOCIDA.")
        mapped_classification = "DESCONOCIDA"
    else: 
        logger.warning(f"Clasificación BAML original no reconocida '{original_baml_classification}'. Mapeando a DESCONOCIDA.")
        mapped_classification = "DESCONOCIDA"
    
    logger.info(f"Clasificación BAML original: '{original_baml_classification}', Mapeada a: '{mapped_classification}'")
    return mapped_classification

# --- Función de Orquestación Principal para este Módulo (actualizada) ---
async def obtener_clasificacion_baml(pixel_array_visual: np.ndarray) -> str:
    """
    Función de alto nivel para convertir píxeles y clasificarlos con BAML.
    Utiliza tenacity para reintentos en la llamada síncrona a BAML.
    """
    if pixel_array_visual is None:
        logger.error("pixel_array_visual es None en obtener_clasificacion_baml.")
        return "ErrorPixelArrayNulo"
        
    if b is None or BamlImage is None: # Comprobación temprana
        logger.critical("Cliente BAML ('b') o BamlImage no disponibles (fallo de importación). No se puede clasificar.")
        return "ErrorBAMLClienteNoDisp"
        
    logger.debug("Convirtiendo pixel_array a base64 para BAML...")
    imagen_base64 = _convert_pixel_array_to_png_base64(pixel_array_visual)
    
    if not imagen_base64:
        logger.error("Fallo al convertir imagen a base64 para BAML.")
        return "ErrorConversionBase64"

    try:
        logger.debug("Llamando a _clasificar_con_baml_tenacity en executor...")
        loop = asyncio.get_event_loop()
        # La función _clasificar_con_baml_tenacity ahora maneja los reintentos internamente.
        clasificacion_final = await loop.run_in_executor(None, _clasificar_con_baml_tenacity, imagen_base64)
        return clasificacion_final
    except BamlClientHttpError as http_error_final: # Si tenacity agota reintentos y relanza la última excepción
        logger.error(f"Error HTTP de BAML después de todos los reintentos: {http_error_final.status_code} - {http_error_final.message}", exc_info=True)
        if http_error_final.status_code == 429:
            return "ErrorBAMLRATE_LIMIT_Final"
        return "ErrorBAMLHTTP_Final"
    except Exception as e_exec: # Otras excepciones no capturadas por tenacity
        logger.exception("Error ejecutando clasificación BAML con tenacity en executor.")
        return "ErrorBAMLExecutorGeneral"


# ... (Función _test_main_real_baml y bloque if __name__ == '__main__' como estaban antes,
#      pero _test_main_real_baml ahora llamará a la versión con tenacity) ...

async def _test_main_real_baml():
    if b is None or BamlImage is None:
        print("Cliente BAML ('b' o BamlImage) no disponible. Test abortado.")
        return
    # Simular un pixel array
    test_array = np.zeros((64, 64), dtype=np.uint8) # Imagen negra simple para prueba
    np.fill_diagonal(test_array, 200) # Añadir algo de estructura
    
    print("Iniciando prueba de clasificación BAML REAL (con tenacity y mapeo)...")
    clasificacion = await obtener_clasificacion_baml(test_array)
    print(f"Clasificación BAML REAL (mapeada) obtenida: '{clasificacion}'")

if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s [%(threadName)s] - %(levelname)s - %(message)s')
    # Descomenta para probar la clasificación con tenacity
    # asyncio.run(_test_main_real_baml())
    logger.info("baml_classification.py cargado (CON tenacity y mapeo a FDT/MTF/BC/DESCONOCIDA).")