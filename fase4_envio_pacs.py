# fase4_envio_pacs.py

import asyncio
import logging
from pathlib import Path

# --- Importaciones del proyecto ---
try:
    import config
    import pacs_operations
    from utils import configurar_logging_aplicacion
except ImportError as e:
    print(f"Error CRÍTICO importando módulos del proyecto: {e}. Asegúrate de que todos los ficheros .py necesarios estén presentes.")
    exit()

# --- Configuración de Logging ---
log_file_path = getattr(config, 'BASE_PROJECT_DIR', Path(".")) / getattr(config, 'LOG_FILENAME', "dicom_workflow.log")
configurar_logging_aplicacion(log_file_path=str(log_file_path), level=config.LOG_LEVEL, log_format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Configuración de Rutas ---
INPUT_DIR = Path(getattr(config, 'BASE_PROJECT_DIR', Path.cwd())) / "f3_linealizados"

async def ejecutar_fase4_envio_pacs():
    """
    Orquestador principal para la fase de envío a PACS.
    """
    logger.info("===== INICIO FASE 4: ENVÍO DE FICHEROS PROCESADOS A PACS =====")

    if not INPUT_DIR.is_dir():
        logger.error(f"El directorio de entrada '{INPUT_DIR}' no existe o no es un directorio.")
        logger.error("Asegúrate de haber ejecutado la fase 3 correctamente.")
        return

    files_to_send = list(INPUT_DIR.rglob("*.dcm"))
    if not files_to_send:
        logger.warning(f"No se encontraron ficheros DICOM (.dcm) para enviar en '{INPUT_DIR}'. Finalizando fase.")
        return

    logger.info(f"Se encontraron {len(files_to_send)} ficheros en '{INPUT_DIR}' listos para ser enviados.")

    try:
        pacs_config_dict = {
            "PACS_IP": config.PACS_IP,
            "PACS_PORT": config.PACS_PORT,
            "PACS_AET": config.PACS_AET,
            "AE_TITLE": config.CLIENT_AET
        }
        logger.info(f"Conectando a PACS: AET='{pacs_config_dict['PACS_AET']}', IP='{pacs_config_dict['PACS_IP']}', Puerto='{pacs_config_dict['PACS_PORT']}'")
        logger.info(f"AE Title del cliente: '{pacs_config_dict['AE_TITLE']}'")
    except AttributeError as e:
        logger.critical(f"Error: falta una variable de configuración de PACS en tu fichero 'config.py'. Detalle: {e}")
        return

    # --- CORRECCIÓN AQUÍ ---
    # Se eliminan los argumentos por palabra clave ('folder_path=' y 'pacs_config=')
    # y se pasan los argumentos por posición, como en main.py.
    all_sent_successfully = await pacs_operations.send_dicom_folder_async(
        str(INPUT_DIR),
        pacs_config_dict
    )

    if all_sent_successfully:
        logger.info("✅ Todos los ficheros procesados han sido enviados a PACS exitosamente.")
    else:
        logger.error("Algunos ficheros pudieron no haberse enviado correctamente. Revisa los logs anteriores para más detalles.")

    logger.info("===== FIN FASE 4: ENVÍO A PACS COMPLETADO =====")


if __name__ == "__main__":
    try:
        asyncio.run(ejecutar_fase4_envio_pacs())
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario.")
    except Exception as e:
        logger.critical(f"Ha ocurrido un error inesperado durante la ejecución de la fase 4: {e}", exc_info=True)