# config.py
import logging
from pathlib import Path

# --- Rutas del Sistema de Ficheros ---
# Es buena práctica definir las rutas relativas al script de configuración o a una base del proyecto.
# Aquí, asumimos que config.py está en el directorio raíz 'pacs/'.
BASE_PROJECT_DIR = Path(__file__).resolve().parent

INPUT_DICOM_DIR = BASE_PROJECT_DIR / "input_dicom_files"
OUTPUT_PROCESSED_DICOM_DIR = BASE_PROJECT_DIR / "output_processed_dicom"
LOG_FILENAME = "dicom_workflow.log" # Nombre del fichero de log

# Ruta al fichero CSV para la calibración de la LUT Kerma
PATH_LUT_CALIBRATION_CSV = BASE_PROJECT_DIR / "data" / "linearizacion.csv"


# --- Configuración de la LUT Kerma ---
KERMA_SCALING_FACTOR = 100.0 # Factor de escalado para los valores de Kerma en la LUT


# --- Configuración del PACS ---
# Reemplaza estos valores con los de tu entorno PACS real
PACS_IP = "localhost"  # Dirección IP o hostname de tu servidor PACS
PACS_PORT = 11112                  # Puerto del servidor PACS
PACS_AET = "DCM4CHEE"              # AE Title del servidor PACS (destino)
CLIENT_AET = "MYPYTHONSCU"         # AE Title de esta aplicación cliente (origen)


# --- Configuración de Logging ---
# Puedes definir el nivel de logging global aquí
LOG_LEVEL = logging.INFO  # Opciones: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# --- Configuración de BAML (si es necesario) ---
# SIMULATE_BAML = True # Añade esta flag si quieres controlar la simulación desde aquí


# --- Parámetros de Linealización Física ---
ENABLE_PHYSICAL_LINEALIZATION_PARAMS = False  # <--- ACTIVAR PARA LA PRUEBA

# Ruta al CSV para la linealización física (puede ser el mismo que para la LUT Kerma si es aplicable)
PATH_CSV_LINEALIZACION_FISICA = BASE_PROJECT_DIR / "data" / "linearizacion.csv" # <--- AJUSTA SI ES OTRO FICHERO

# Tipo de RQA por defecto a usar si no se determina de otra forma para la linealización física
DEFAULT_RQA_TYPE_LINEALIZATION = "RQA5" # <--- AJUSTA AL RQA DE TUS DATOS DE CALIBRACIÓN

# Factores RQA (SNR_in^2 / 1000) para diferentes calidades de haz.
# Estos son cruciales para convertir K_uGy a "quanta/area".
# Deberías tener los valores correctos para los RQA que uses.
RQA_FACTORS_PHYSICAL_LINEALIZATION: dict[str, float] = {
    "RQA3": 0.000085, # Ejemplo, reemplaza con tus valores reales
    "RQA5": 0.000123, # Ejemplo, como el de linealize.py
    "RQA7": 0.000250, # Ejemplo
    "RQA9": 0.000456, # Ejemplo
    # ... añade más RQA y sus factores según sea necesario
}

# Private Creator ID para los tags de linealización física en la cabecera DICOM
PRIVATE_CREATOR_ID_LINEALIZATION = "MIAPP_LINFO_V1" # <--- ELIGE UN ID ÚNICO Y SIGNIFICATIVO


# --- Otros Parámetros de la Aplicación ---
DICOM_TAG_FOR_CLASSIFICATION = "ImageComments" 
CLASSIFICATION_TAG_PREFIX = "QC_Class:"


# --- Verificaciones (opcional pero recomendado) ---
def check_paths():
    """Verifica la existencia de directorios y ficheros cruciales."""
    logger_cfg = logging.getLogger(__name__) # Logger local para esta función
    paths_to_check = {
        "Directorio de entrada DICOM": INPUT_DICOM_DIR,
        "Fichero CSV de calibración LUT Kerma": PATH_LUT_CALIBRATION_CSV
    }
    # Añadir el CSV de linealización física a la verificación si está habilitada
    if ENABLE_PHYSICAL_LINEALIZATION_PARAMS:
        paths_to_check["Fichero CSV de linealización física"] = PATH_CSV_LINEALIZACION_FISICA

    all_paths_valid = True
    for description, path_obj in paths_to_check.items():
        if not path_obj.exists():
            logger_cfg.warning(f"CONFIG WARNING: {description} no encontrado en la ruta esperada: {path_obj}")
            all_paths_valid = False
    
    try:
        OUTPUT_PROCESSED_DICOM_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger_cfg.error(f"CONFIG ERROR: No se pudo crear el directorio de salida {OUTPUT_PROCESSED_DICOM_DIR}: {e}")
        all_paths_valid = False
    return all_paths_valid

# Configurar un logger básico si config.py se ejecuta directamente para check_paths
if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    logger_main_cfg = logging.getLogger(__name__)
    if check_paths():
        logger_main_cfg.info("Verificación de rutas en config.py completada con éxito.")
    else:
        logger_main_cfg.error("Algunas rutas configuradas en config.py no son válidas o no existen.")

# --- FIN DE config.py ---