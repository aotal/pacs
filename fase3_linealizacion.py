# fase3_linealizacion.py

import asyncio
import logging
from pathlib import Path
import shutil
from typing import Optional, Dict

# --- Importaciones del proyecto ---
# Se asume que estos ficheros existen y están en la misma carpeta o en el path.
try:
    import config
    import dicom_processing_pipeline
    import linealize
    import pandas as pd
    from utils import configurar_logging_aplicacion
except ImportError as e:
    print(f"Error CRÍTICO importando módulos del proyecto: {e}. Asegúrate de que todos los ficheros .py necesarios estén presentes.")
    exit()

# --- Configuración de Logging ---
log_file_path = getattr(config, 'BASE_PROJECT_DIR', Path(".")) / getattr(config, 'LOG_FILENAME', "dicom_workflow.log")
configurar_logging_aplicacion(log_file_path=str(log_file_path), level=config.LOG_LEVEL, log_format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Configuración de Rutas ---
INPUT_DIR = Path(getattr(config, 'BASE_PROJECT_DIR', Path.cwd())) / "f2_clasificados"
OUTPUT_DIR = Path(getattr(config, 'BASE_PROJECT_DIR', Path.cwd())) / "f3_linealizados"
CLASES = ["FDT", "MTF", "TOR"]

async def procesar_y_linealizar_fichero(
    filepath: Path,
    clasificacion: str,
    lut_kerma_data: tuple,
    calib_df_linealizacion: Optional[pd.DataFrame],
    rqa_type_linealizacion: Optional[str],
    rqa_factors_linealizacion: Optional[Dict[str, float]]
) -> Optional[Path]:
    """
    Aplica la linealización y el procesamiento final a un único fichero DICOM.
    """
    logger.info(f"--- Iniciando Fase 3 para: {filepath.name} (Clase: {clasificacion}) ---")
    try:
        ds = dicom_processing_pipeline.pydicom.dcmread(str(filepath), force=True)
    except Exception as e:
        logger.error(f"No se pudo leer el fichero DICOM {filepath.name}. Error: {e}")
        return None

    # --- Cálculo de la pendiente de linealización (lógica de main.py) ---
    slope_linealizacion: Optional[float] = None
    rqa_type_para_tags: Optional[str] = None

    if getattr(config, 'ENABLE_PHYSICAL_LINEALIZATION_PARAMS', False) and \
       calib_df_linealizacion is not None and \
       rqa_type_linealizacion and rqa_factors_linealizacion:
        
        logger.info(f"Calculando pendiente de linealización para {filepath.name} (RQA: {rqa_type_linealizacion})...")
        slope_linealizacion = linealize.calculate_linearization_slope(
            calibration_df=calib_df_linealizacion,
            rqa_type=rqa_type_linealizacion,
            rqa_factors_dict=rqa_factors_linealizacion
        )
        if slope_linealizacion is not None:
            rqa_type_para_tags = rqa_type_linealizacion
            logger.info(f"Pendiente calculada: {slope_linealizacion:.4e}")
        else:
            logger.warning(f"No se pudo calcular la pendiente para {filepath.name}.")
    else:
        logger.info("Linealización física omitida (desactivada o datos insuficientes en config).")

    # --- Aplicación de LUT Kerma y guardado final (lógica de main.py) ---
    pixel_values_kerma_lut, kerma_values_kerma_lut = lut_kerma_data
    private_creator_id = getattr(config, 'PRIVATE_CREATOR_ID_LINEALIZATION', "MY_APP_LINFO")

    output_filepath = dicom_processing_pipeline.process_and_prepare_dicom_for_pacs(
        ds=ds,
        clasificacion_baml_mapeada=clasificacion, # Usamos la clasificación de la carpeta
        pixel_values_calib=pixel_values_kerma_lut,
        kerma_values_calib=kerma_values_kerma_lut,
        output_base_dir=OUTPUT_DIR,
        original_filename=filepath.name,
        linearization_slope_param=slope_linealizacion,
        rqa_type_param=rqa_type_para_tags,
        private_creator_id_linealizacion=private_creator_id
    )

    if output_filepath:
        logger.info(f"Fichero procesado y guardado en: {output_filepath}")
        return output_filepath
    else:
        logger.error(f"Fallo en el procesamiento final de {filepath.name}.")
        return None


async def ejecutar_fase3():
    """
    Orquestador principal para la fase de linealización.
    """
    logger.info("===== INICIO FASE 3: LINEALIZACIÓN Y PROCESAMIENTO FINAL =====")

    # --- Preparar directorios ---
    if not INPUT_DIR.exists():
        logger.error(f"El directorio de entrada '{INPUT_DIR}' no existe. Ejecuta la fase 2 primero.")
        return

    if OUTPUT_DIR.exists():
        logger.warning(f"Eliminando directorio de salida existente: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Cargar datos de calibración (lógica de main.py) ---
    logger.info(f"Cargando datos LUT Kerma desde: {config.PATH_LUT_CALIBRATION_CSV}")
    pixel_cal_kerma, kerma_cal_kerma = dicom_processing_pipeline.load_calibration_data(
        str(config.PATH_LUT_CALIBRATION_CSV))
    if pixel_cal_kerma is None or kerma_cal_kerma is None:
        logger.critical("No se pudieron cargar los datos de calibración para LUT Kerma. Abortando.")
        return
    lut_kerma_data = (pixel_cal_kerma, kerma_cal_kerma)

    df_calib_lin: Optional[pd.DataFrame] = None
    rqa_type_global: Optional[str] = None
    rqa_factors_global: Optional[Dict[str, float]] = None

    if getattr(config, 'ENABLE_PHYSICAL_LINEALIZATION_PARAMS', False):
        path_csv_lin = str(getattr(config, 'PATH_CSV_LINEALIZACION_FISICA', config.PATH_LUT_CALIBRATION_CSV))
        logger.info(f"Cargando datos de linealización desde: {path_csv_lin}")
        df_calib_lin = linealize.obtener_datos_calibracion_vmp_k_linealizacion(path_csv_lin)
        if df_calib_lin is not None:
            rqa_type_global = getattr(config, 'DEFAULT_RQA_TYPE_LINEALIZATION', "RQA5")
            rqa_factors_global = getattr(config, 'RQA_FACTORS_PHYSICAL_LINEALIZATION', {})
            logger.info(f"Datos de linealización cargados. Se usará RQA por defecto: {rqa_type_global}")
        else:
            logger.warning("No se pudieron cargar datos para la linealización física.")

    # --- Procesar ficheros por cada clase ---
    tasks = []
    for clase in CLASES:
        clase_dir = INPUT_DIR / clase
        if not clase_dir.is_dir():
            logger.warning(f"No se encontró el directorio para la clase '{clase}'. Omitiendo.")
            continue
        
        dicom_files = list(clase_dir.glob("*.dcm"))
        logger.info(f"Encontrados {len(dicom_files)} ficheros en la carpeta '{clase}'.")
        
        for fp in dicom_files:
            tasks.append(procesar_y_linealizar_fichero(
                filepath=fp,
                clasificacion=clase,
                lut_kerma_data=lut_kerma_data,
                calib_df_linealizacion=df_calib_lin,
                rqa_type_linealizacion=rqa_type_global,
                rqa_factors_linealizacion=rqa_factors_global
            ))

    if not tasks:
        logger.warning("No se encontraron ficheros para procesar.")
        return

    results = await asyncio.gather(*tasks)
    success_count = sum(1 for r in results if r is not None)
    
    logger.info(f"===== FIN FASE 3: {success_count} de {len(tasks)} ficheros procesados correctamente. =====")


if __name__ == "__main__":
    asyncio.run(ejecutar_fase3())