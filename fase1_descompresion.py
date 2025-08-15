# fase1_descompresion.py (VERSIÓN ORDENADA CRONOLÓGICAMENTE)

import asyncio
import logging
from pathlib import Path
import shutil
import pydicom
from pydicom.filewriter import dcmwrite
from pydicom.uid import ExplicitVRLittleEndian
from datetime import datetime # ¡NUEVO! Importación para manejar fechas y horas

# Se asume que tienes los ficheros config.py y utils.py simplificados
import config
from utils import configurar_logging_aplicacion, clean_filename_part

# Configurar logging
configurar_logging_aplicacion(
    log_file_path=str(config.BASE_PROJECT_DIR / config.LOG_FILENAME),
    level=config.LOG_LEVEL,
    log_format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# --- Rutas ---
INPUT_DIR = config.INPUT_DIR
OUTPUT_DIR = config.OUTPUT_DIR

def is_dicom_file(filepath: Path) -> bool:
    """Comprueba si un fichero es un DICOM válido por su contenido."""
    if filepath.stat().st_size < 132: return False
    try:
        with open(filepath, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False

async def descomprimir_y_renombrar_fichero(filepath: Path, file_index: int):
    """Lee, descomprime y guarda un DICOM con un nuevo nombre usando un índice secuencial."""
    # (El contenido de esta función no cambia)
    logger.info(f"Procesando fichero: {filepath.name} (Índice cronológico: {file_index})")
    try:
        ds = pydicom.dcmread(str(filepath), force=True)
        if ds.file_meta.TransferSyntaxUID.is_compressed:
            ds.decompress()
        
        detector_id_fn = ds.get('DetectorID', 'NoDetID')
        kvp_fn = ds.get('KVP', 'NoKVP')
        exposure_uas_fn = float(ds.get('ExposureInuAs', 0.0))
        exposure_index_fn = ds.get('ExposureIndex', 'NoIE')

        new_filename_base = (
            f"Img{file_index}"
            f"_{clean_filename_part(detector_id_fn)}"
            f"_KVP{clean_filename_part(kvp_fn)}"
            f"_mAs{round(exposure_uas_fn / 1000.0, 2)}"
            f"_IE{clean_filename_part(exposure_index_fn)}"
            f"_{ds.SOPInstanceUID}"
        )
        new_filename = new_filename_base[:200] + ".dcm"
        output_filepath = OUTPUT_DIR / new_filename
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dcmwrite(str(output_filepath), ds)
        
        logger.info(f" -> Guardado como: {new_filename}")
        return True
    except Exception as e:
        logger.error(f"Error procesando '{filepath.name}': {e}", exc_info=True)
        return False

async def ejecutar_fase1():
    logger.info("===== INICIO FASE 1: DESCOMPRESIÓN Y RENOMBRADO (ORDENADO CRONOLÓGICAMENTE) =====")
    
    INPUT_DIR.mkdir(exist_ok=True)
    
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- ¡NUEVO! Lógica de ordenación ---
    # 1. Encontrar todos los ficheros DICOM válidos
    logger.info("Buscando y leyendo metadatos de los ficheros DICOM...")
    all_dicom_files = [f for f in INPUT_DIR.rglob('*') if f.is_file() and is_dicom_file(f)]
    
    files_with_timestamps = []
    files_without_timestamps = []

    # 2. Leer la fecha y hora de cada fichero
    for fp in all_dicom_files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True) # Leemos solo la cabecera
            acq_date = ds.get("AcquisitionDate", "")
            acq_time = ds.get("AcquisitionTime", "")
            
            if acq_date and acq_time:
                # Combinamos fecha y hora. Manejamos segundos fraccionales.
                time_str = f"{acq_date}{acq_time.split('.')[0]}"
                timestamp = datetime.strptime(time_str, "%Y%m%d%H%M%S")
                files_with_timestamps.append((timestamp, fp))
            else:
                logger.warning(f"El fichero '{fp.name}' no tiene etiquetas de fecha/hora. Se procesará al final.")
                files_without_timestamps.append(fp)
        except Exception as e:
            logger.error(f"No se pudo leer la cabecera de '{fp.name}': {e}. Se procesará al final.")
            files_without_timestamps.append(fp)

    # 3. Ordenar la lista de ficheros por su timestamp
    files_with_timestamps.sort(key=lambda item: item[0])
    
    # Reconstruir la lista de ficheros, ahora en orden cronológico
    sorted_files = [fp for ts, fp in files_with_timestamps] + files_without_timestamps
    
    if not sorted_files:
        logger.warning(f"No se encontraron ficheros DICOM válidos para procesar.")
        return

    logger.info(f"Se encontraron y ordenaron {len(sorted_files)} ficheros DICOM válidos para procesar.")
    
    # 4. Crear las tareas de procesamiento con la lista ya ordenada
    tasks = [
        descomprimir_y_renombrar_fichero(filepath, index) 
        for index, filepath in enumerate(sorted_files, 1)
    ]
    
    results = await asyncio.gather(*tasks)
    
    success_count = sum(1 for r in results if r)
    logger.info(f"===== FIN FASE 1: {success_count} de {len(sorted_files)} ficheros procesados correctamente. =====")

if __name__ == "__main__":
    asyncio.run(ejecutar_fase1())