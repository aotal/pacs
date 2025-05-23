# main.py
import asyncio
import logging
from pathlib import Path
import shutil 
from typing import Any, Optional, Dict, List

try:
    import config
    from utils import configurar_logging_aplicacion
except ImportError as e_imp:
    print(f"Error CRÍTICO importando config o utils: {e_imp}.")
    exit()

log_file_path_config = getattr(config, 'BASE_PROJECT_DIR', Path(".")) / getattr(config, 'LOG_FILENAME', "dicom_workflow.log")
log_level_config = getattr(config, 'LOG_LEVEL', logging.INFO)
log_format_config = getattr(config, 'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
configurar_logging_aplicacion(log_file_path=str(log_file_path_config), level=log_level_config, log_format=log_format_config)
logger = logging.getLogger(__name__)

try:
    import dicom_processing_pipeline
    import baml_classification # Esta versión DEBE devolver el valor MAPEADO
    import linealize 
    import pacs_operations
    import pandas as pd 
except ImportError as e_imp_modules:
    logger.critical(f"Error CRÍTICO importando módulos de procesamiento: {e_imp_modules}. ", exc_info=True)
    exit()

if hasattr(config, 'check_paths') and callable(config.check_paths):
    if not config.check_paths():
        logger.critical("Abortando debido a errores de configuración de rutas.")
        exit()
else:
    try:
        config.INPUT_DICOM_DIR.mkdir(parents=True, exist_ok=True)
        config.OUTPUT_PROCESSED_DICOM_DIR.mkdir(parents=True, exist_ok=True)
        if not config.PATH_LUT_CALIBRATION_CSV.is_file():
            logger.warning(f"CONFIG WARNING: Fichero CSV de calibración LUT Kerma no encontrado: {config.PATH_LUT_CALIBRATION_CSV}")
    except AttributeError as e_attr_config:
        logger.critical(f"Error accediendo a rutas de config: {e_attr_config}. Verifica config.py.")
        exit()
    except Exception as e_path_create:
        logger.critical(f"Error creando directorios: {e_path_create}")
        exit()

async def process_single_dicom_file(
    dicom_filepath: Path,
    lut_kerma_calib_data: tuple, 
    lut_linealizacion_calib_df: Optional[pd.DataFrame], 
    rqa_type_for_linealizacion: Optional[str],
    rqa_factors_for_linealizacion: Optional[Dict[str, float]]
) -> Optional[Path]:
    file_basename = dicom_filepath.name
    logger.info(f"--- Iniciando procesamiento para: {file_basename} ---")
    ds, pixel_array_for_baml = await dicom_processing_pipeline.read_and_decompress_dicom(dicom_filepath)
    if ds is None or pixel_array_for_baml is None:
        logger.error(f"No se pudo leer o obtener píxeles de {file_basename}. Omitiendo.")
        return None

    logger.info(f"Enviando imagen {file_basename} para clasificación BAML...")
    # baml_classification.obtener_clasificacion_baml ahora devuelve el valor mapeado
    clasificacion_baml_valor_mapeado = await baml_classification.obtener_clasificacion_baml(pixel_array_for_baml)
    logger.info(f"Clasificación BAML (mapeada) para {file_basename}: '{clasificacion_baml_valor_mapeado}'")
    
    clasificacion_a_guardar_final = "ClasificacionFallida" # Default si BAML da error
    if clasificacion_baml_valor_mapeado and not clasificacion_baml_valor_mapeado.startswith("Error"):
        clasificacion_a_guardar_final = clasificacion_baml_valor_mapeado
    else:
        logger.warning(f"Clasificación BAML para {file_basename} indicó problema o era inválida: '{clasificacion_baml_valor_mapeado}'. Usando '{clasificacion_a_guardar_final}'.")

    slope_linealizacion: Optional[float] = None
    rqa_type_actual_para_tags: Optional[str] = None
    if getattr(config, 'ENABLE_PHYSICAL_LINEALIZATION_PARAMS', False) and \
       lut_linealizacion_calib_df is not None and \
       rqa_type_for_linealizacion and rqa_factors_for_linealizacion:
        logger.info(f"Calculando pendiente de linealización física para {file_basename} (RQA: {rqa_type_for_linealizacion})...")
        slope_linealizacion = linealize.calculate_linearization_slope(
            calibration_df=lut_linealizacion_calib_df,
            rqa_type=rqa_type_for_linealizacion,
            rqa_factors_dict=rqa_factors_for_linealizacion )
        if slope_linealizacion is not None:
            rqa_type_actual_para_tags = rqa_type_for_linealizacion
            logger.info(f"Pendiente de linealización física calculada para {file_basename}: {slope_linealizacion:.4e}")
        else:
            logger.warning(f"No se pudo calcular la pendiente de linealización física para {file_basename}.")
    else:
        logger.info(f"Linealización física omitida para {file_basename} (desactivada o datos insuficientes).")

    logger.info(f"Aplicando modificaciones finales y LUT Kerma a {file_basename}...")
    pixel_values_kerma_lut, kerma_values_kerma_lut = lut_kerma_calib_data
    private_creator_lin_id = getattr(config, 'PRIVATE_CREATOR_ID_LINEALIZATION', "MY_APP_LINFO_DEFAULT") # Asegurar fallback

    output_dicom_filepath = dicom_processing_pipeline.process_and_prepare_dicom_for_pacs(
        ds=ds, 
        clasificacion_baml_mapeada=clasificacion_a_guardar_final, # Nombre de parámetro consistente
        pixel_values_lut_calib=pixel_values_kerma_lut,
        kerma_values_lut_calib=kerma_values_kerma_lut,
        kerma_scaling_factor_lut=config.KERMA_SCALING_FACTOR,
        output_base_dir=config.OUTPUT_PROCESSED_DICOM_DIR,
        original_filename=file_basename,
        linearization_slope_param=slope_linealizacion, 
        rqa_type_param=rqa_type_actual_para_tags, 
        private_creator_id_linealizacion=private_creator_lin_id 
    )
    if output_dicom_filepath:
        logger.info(f"Fichero {file_basename} procesado y guardado como: {output_dicom_filepath.name}")
        return output_dicom_filepath
    else:
        logger.error(f"Fallo en el procesamiento final y guardado de {file_basename}.")
        return None

async def main_orchestrator():
    logger.info("===== INICIO DEL WORKFLOW DE PROCESAMIENTO DICOM (CON BAML MAPEADO Y SOBRESCRITO) =====")
    if hasattr(baml_classification, 'b') and baml_classification.b is None and \
       not getattr(config, 'SIMULATE_BAML', False): 
        logger.critical("Cliente BAML no disponible y simulación no activada. Abortando.")
        return
    
    logger.info(f"Cargando datos de calibración para LUT Kerma desde: {config.PATH_LUT_CALIBRATION_CSV}")
    pixel_cal_kerma, kerma_cal_kerma = dicom_processing_pipeline.load_kerma_calibration_data_for_lut(
        str(config.PATH_LUT_CALIBRATION_CSV) )
    if pixel_cal_kerma is None or kerma_cal_kerma is None:
        logger.critical("No se pudieron cargar los datos de calibración para la LUT Kerma. Abortando workflow.")
        return
    lut_kerma_data_tuple = (pixel_cal_kerma, kerma_cal_kerma)

    df_calib_linealizacion_fisica: Optional[pd.DataFrame] = None
    rqa_type_para_linealizacion_global: Optional[str] = None
    rqa_factors_dict_global: Optional[Dict[str, float]] = None
    if getattr(config, 'ENABLE_PHYSICAL_LINEALIZATION_PARAMS', False):
        path_csv_lin_fisica_str = str(getattr(config, 'PATH_CSV_LINEALIZACION_FISICA', config.PATH_LUT_CALIBRATION_CSV))
        logger.info(f"Cargando datos de calibración para Linealización Física desde: {path_csv_lin_fisica_str}")
        df_calib_linealizacion_fisica = linealize.obtener_datos_calibracion_vmp_k_linealizacion(
             path_csv_lin_fisica_str )
        if df_calib_linealizacion_fisica is None:
            logger.warning("No se pudieron cargar datos para la linealización física.")
        else:
            rqa_type_para_linealizacion_global = getattr(config, 'DEFAULT_RQA_TYPE_LINEALIZATION', "RQA5")
            rqa_factors_dict_global = getattr(config, 'RQA_FACTORS_PHYSICAL_LINEALIZATION', 
                                             getattr(linealize, 'RQA_FACTORS_EXAMPLE', {})) 
            if not rqa_factors_dict_global:
                 logger.warning("No se encontraron RQA_FACTORS para linealización física.")
            logger.info(f"Se usarán datos para linealización física con RQA: {rqa_type_para_linealizacion_global}")
    else:
        logger.info("Cálculo y almacenamiento de parámetros de linealización física está DESACTIVADO.")

    input_dir_path = Path(config.INPUT_DICOM_DIR)
    dicom_files_to_process = [f for f in input_dir_path.iterdir() if f.is_file()] 
    if not dicom_files_to_process:
        logger.info(f"No se encontraron ficheros en {input_dir_path}. Finalizando.")
        return
    logger.info(f"Se encontraron {len(dicom_files_to_process)} ficheros para procesar en {input_dir_path}.")

    tasks = [process_single_dicom_file(
        dicom_filepath=fp, lut_kerma_calib_data=lut_kerma_data_tuple,
        lut_linealizacion_calib_df=df_calib_linealizacion_fisica, 
        rqa_type_for_linealizacion=rqa_type_para_linealizacion_global,
        rqa_factors_for_linealizacion=rqa_factors_dict_global
    ) for fp in dicom_files_to_process]
            
    processed_output_paths: List[Path] = []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result_item in enumerate(results):
        original_file_path_obj = dicom_files_to_process[i]
        if isinstance(result_item, Path) and result_item.exists():
            processed_output_paths.append(result_item)
            logger.info(f"ÉXITO en procesamiento de {original_file_path_obj.name} -> {result_item.name}")
        elif isinstance(result_item, Exception):
            logger.error(f"ERROR EXCEPCIÓN procesando {original_file_path_obj.name}: {result_item}", exc_info=result_item)
        elif result_item is None: 
            logger.warning(f"FALLO/OMISIÓN: {original_file_path_obj.name} no generó fichero de salida.")
        else: 
            logger.error(f"RESULTADO INESPERADO para {original_file_path_obj.name}: {result_item}")

    valid_processed_files_to_send = [p for p in processed_output_paths if p.exists()]
    if valid_processed_files_to_send:
        logger.info(f"Se procesaron {len(valid_processed_files_to_send)} ficheros con éxito. Iniciando envío a PACS...")
        pacs_config_dict = {
            "PACS_IP": config.PACS_IP, "PACS_PORT": config.PACS_PORT,
            "PACS_AET": config.PACS_AET, "AE_TITLE": config.CLIENT_AET
        }
        all_sent_successfully = await pacs_operations.send_dicom_folder_async(
            str(config.OUTPUT_PROCESSED_DICOM_DIR), pacs_config_dict ) 
        if all_sent_successfully: logger.info("Todos los ficheros procesados enviados a PACS exitosamente.")
        else: logger.warning("Algunos ficheros procesados pudieron no haberse enviado a PACS.")
    else:
        logger.info("No se generaron ficheros de salida válidos, no se enviará nada al PACS.")
    logger.info("===== FIN DEL WORKFLOW DE PROCESAMIENTO DICOM (CON BAML MAPEADO Y SOBRESCRITO) =====")

if __name__ == "__main__":
    logger.info(f"Iniciando aplicación desde: {Path(__file__).name}")
    try:
        asyncio.run(main_orchestrator())
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario.")
    except Exception as e_global:
        logger.critical(f"Error global no capturado: {e_global}", exc_info=True)
    finally:
        logger.info("Aplicación finalizada.")