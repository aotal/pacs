# dicom_processing_pipeline.py
import logging
import os
from pathlib import Path
from typing import Tuple, Optional, Any, Dict 

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, DataElement 
from pydicom.errors import InvalidDicomError
from pydicom.sequence import Sequence
from pydicom.filewriter import dcmwrite
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pydicom.tag import Tag

import shutil

try:
    import config 
    from utils import get_translated_location, clean_filename_part
    import linealize 
except ImportError as e_pipeline_import:
    logging.critical(f"Error CRÍTICO importando módulos necesarios (utils, config, linealize) en dicom_processing_pipeline.py: {e_pipeline_import}. ")
    raise 

logger = logging.getLogger(__name__)

def load_calibration_data(csv_filepath: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads calibration data (Pixel Value vs. Physical Value) from a CSV file.
    """
    try:
        path_obj = Path(csv_filepath)
        if not path_obj.is_file():
            logger.error(f"Fichero CSV de calibración no encontrado: {csv_filepath}")
            return None, None
        df = pd.read_csv(path_obj)
        required_cols = ['VMP', 'K_uGy'] 
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"El CSV {csv_filepath} para calibración debe contener las columnas: {missing}.")
            return None, None
        if df[required_cols].isnull().any().any():
            logger.warning(f"Valores nulos encontrados en columnas {required_cols} de {csv_filepath}. Se eliminarán esas filas.")
            df.dropna(subset=required_cols, inplace=True)
            if df.empty:
                logger.error(f"El CSV {csv_filepath} quedó vacío después de eliminar filas con NaNs.")
                return None, None
        pixel_values = df['VMP'].to_numpy(dtype=float)
        kerma_values = df['K_uGy'].to_numpy(dtype=float)
        if len(pixel_values) < 2: 
            logger.error(f"No hay suficientes datos válidos (se necesitan al menos 2 puntos) en {csv_filepath} para la calibración.")
            return None, None
        logger.info(f"Datos de calibración cargados desde {csv_filepath}: {len(pixel_values)} puntos.")
        return pixel_values, kerma_values
    except Exception as e:
        logger.error(f"Error leyendo el fichero CSV de calibración ({csv_filepath}): {e}", exc_info=True)
        return None, None

async def read_and_decompress_dicom(filepath: Path) -> tuple[Optional[Dataset], Optional[Any]]:
    logger.info(f"Leyendo y (si es necesario) descomprimiendo: {filepath.name}")
    try:
        if not filepath.is_file():
            logger.error(f"El archivo no existe: {filepath}")
            return None, None
        ds = pydicom.dcmread(str(filepath), force=True)
        if hasattr(ds, 'file_meta') and \
           hasattr(ds.file_meta, 'TransferSyntaxUID') and \
           ds.file_meta.TransferSyntaxUID.is_compressed:
            try:
                logger.info(f"Archivo {filepath.name} está comprimido ({ds.file_meta.TransferSyntaxUID.name}). Descomprimiendo...")
                ds.decompress() 
                logger.info(f"Archivo {filepath.name} descomprimido exitosamente.")
            except Exception as e_decompress:
                logger.error(f"Error al descomprimir {filepath.name}: {e_decompress}. ")
        pixel_array_for_baml = ds.pixel_array 
        logger.info(f"Archivo DICOM leído y pixel_array (para BAML) obtenido de: {filepath.name}")
        return ds, pixel_array_for_baml
    except InvalidDicomError:
        logger.error(f"Archivo inválido o no es DICOM: {filepath.name}")
        return None, None
    except AttributeError as ae: 
        logger.error(f"Atributo no encontrado procesando {filepath.name}: {ae}")
        return None, None
    except Exception as e: 
        logger.error(f"Error general leyendo o descomprimiendo {filepath.name}: {e}", exc_info=True)
        return None, None

def _apply_rescale_tags_for_linearization(
    ds: Dataset, 
    pixel_values_calibration: np.ndarray, 
    kerma_values_calibration: np.ndarray
) -> Dataset:
    """
    Calculates and applies Rescale Slope and Intercept for linearization based on calibration data.
    This method replaces the Modality LUT approach.
    """
    sop_uid = ds.SOPInstanceUID if 'SOPInstanceUID' in ds else 'UID_DESCONOCIDO'
    logger.info(f"[{sop_uid}] Aplicando linealización mediante Rescale Slope/Intercept.")

    # 1. Perform linear regression to find slope and intercept
    if not (isinstance(pixel_values_calibration, np.ndarray) and isinstance(kerma_values_calibration, np.ndarray)):
        raise TypeError("pixel_values_calibration y kerma_values_calibration deben ser numpy arrays.")
    if len(pixel_values_calibration) < 2:
        logger.error(f"[{sop_uid}] No hay suficientes puntos de calibración (<2) para calcular Rescale Slope/Intercept.")
        return ds

    # Ensure calibration data is sorted by pixel value for polyfit
    sort_indices = np.argsort(pixel_values_calibration)
    sorted_pixel_values = pixel_values_calibration[sort_indices]
    sorted_kerma_values = kerma_values_calibration[sort_indices]

    slope, intercept = np.polyfit(sorted_pixel_values, sorted_kerma_values, 1)
    
    # 2. Apply new values to the dataset
    ds.RescaleSlope = float(slope)
    ds.RescaleIntercept = float(intercept)
    ds.RescaleType = 'K_uGy' # Set a descriptive unit for the output of the rescale operation
    
    logger.info(f"[{sop_uid}] Nuevos valores aplicados: RescaleSlope={ds.RescaleSlope:.6f}, RescaleIntercept={ds.RescaleIntercept:.6f}, RescaleType={ds.RescaleType}")

    # 3. Remove Modality LUT Sequence to avoid conflicting transformations
    if 'ModalityLUTSequence' in ds:
        del ds.ModalityLUTSequence
        logger.info(f"[{sop_uid}] ModalityLUTSequence eliminada para usar Rescale Slope/Intercept.")
        
    # 4. Update Window Center/Width for a reasonable initial display based on rescaled values
    # Get the full pixel value range from the original image
    bits_stored = ds.BitsStored
    pixel_representation = ds.PixelRepresentation
    
    if pixel_representation == 1:  # Signed
        min_pixel_val = -(2**(bits_stored - 1))
        max_pixel_val = (2**(bits_stored - 1)) - 1
    else:  # Unsigned
        min_pixel_val = 0
        max_pixel_val = (2**bits_stored) - 1
        
    min_output_val = min_pixel_val * ds.RescaleSlope + ds.RescaleIntercept
    max_output_val = max_pixel_val * ds.RescaleSlope + ds.RescaleIntercept

    # Ensure window width is at least 1
    new_ww = max_output_val - min_output_val
    if new_ww < 1.0: new_ww = 1.0
    
    new_wc = min_output_val + new_ww / 2.0
    
    ds.WindowCenter = float(new_wc)
    ds.WindowWidth = float(new_ww)
    logger.debug(f"[{sop_uid}] Nuevos WC/WW: {ds.WindowCenter}/{ds.WindowWidth} basados en valores re-escalados.")

    return ds

def _apply_modality_lut_manual(pixel_array: np.ndarray, ds: Dataset) -> np.ndarray:
    """
    Manually applies the Modality LUT or Rescale transformation to the pixel array.
    This function is a replacement for pydicom.pixels.apply_modality_lut.
    """
    if 'ModalityLUTSequence' in ds and len(ds.ModalityLUTSequence) > 0:
        # This part is now legacy, but kept for reference
        logger.info("Applying Modality LUT transformation (manual fallback).")
        item = ds.ModalityLUTSequence[0]
        lut_descriptor = item.LUTDescriptor
        first_mapped_value = lut_descriptor[1]
        lut_data = np.frombuffer(item.LUTData, dtype=np.uint16)
        pixel_array_float = pixel_array.astype(np.float64)
        interpolated_pixels = np.interp(
            pixel_array_float,
            np.arange(first_mapped_value, first_mapped_value + len(lut_data)),
            lut_data
        )
        return interpolated_pixels.astype(np.float64)
    
    elif 'RescaleIntercept' in ds and 'RescaleSlope' in ds:
        logger.info("Applying Rescale Slope and Intercept transformation (manual fallback).")
        return pixel_array.astype(np.float64) * ds.RescaleSlope + ds.RescaleIntercept
    else:
        logger.info("No Modality LUT or Rescale Slope/Intercept found. Returning original pixel array.")
        return pixel_array.astype(np.float64)

def process_and_prepare_dicom_for_pacs(
    ds: Dataset, 
    clasificacion_baml_mapeada: Optional[str],
    pixel_values_calib: np.ndarray,
    kerma_values_calib: np.ndarray,
    output_base_dir: Path,
    original_filename: str,
    linearization_slope_param: Optional[float] = None, 
    rqa_type_param: Optional[str] = None,
    private_creator_id_linealizacion: Optional[str] = None
) -> Optional[Path]:
    sop_uid = ds.SOPInstanceUID if 'SOPInstanceUID' in ds else f"Original_{original_filename}"
    logger.info(f"Procesando dataset para PACS: {sop_uid}")
    
    try:
        # 1. Modificar PatientID y PatientName
        detector_id_val = ds.get('DetectorID', None) 
        ds.PatientID = str(detector_id_val) if detector_id_val is not None else "UnknownDetectorID"
        logger.info(f"[{sop_uid}] PatientID establecido/actualizado a: {ds.PatientID}")

        station_name_val = ds.get('StationName', "UnknownStation")
        private_tag_val_bytes_or_str = ds.get((0x200B, 0x7096), None)
        location_original = private_tag_val_bytes_or_str.value if private_tag_val_bytes_or_str else None
        translated_location = get_translated_location(location_original) 
        ds.PatientName = f"{str(station_name_val)}_{translated_location}"
        logger.info(f"[{sop_uid}] PatientName establecido/actualizado a: {ds.PatientName}")

        # --- INICIO: Lógica para clasificación BAML ---
        tag_keyword_clasificacion = getattr(config, 'DICOM_TAG_FOR_CLASSIFICATION', 'ImageComments')
        valor_a_escribir_clasificacion = None

        if clasificacion_baml_mapeada and \
           not clasificacion_baml_mapeada.startswith("Error") and \
           clasificacion_baml_mapeada not in ["BAML_OTRO", "ClasificacionFallida", "Desconocida", "ErrorPixelArrayNulo"]:
            valor_a_escribir_clasificacion = clasificacion_baml_mapeada
        elif clasificacion_baml_mapeada: 
             logger.warning(f"[{sop_uid}] Clasificación BAML mapeada fue '{clasificacion_baml_mapeada}' (error o placeholder). No se actualizará '{tag_keyword_clasificacion}'.")
        else: 
            logger.warning(f"[{sop_uid}] No se proporcionó clasificación BAML mapeada válida. No se actualizará '{tag_keyword_clasificacion}'.")

        if valor_a_escribir_clasificacion:
            logger.info(f"[{sop_uid}] Estableciendo (sobrescribiendo) '{tag_keyword_clasificacion}' a: '{valor_a_escribir_clasificacion}'")
            try:
                if hasattr(ds, tag_keyword_clasificacion) or tag_keyword_clasificacion in pydicom.datadict.keyword_dict:
                    setattr(ds, tag_keyword_clasificacion, valor_a_escribir_clasificacion)
                else:
                    tag_address = Tag(tag_keyword_clasificacion)
                    vr = pydicom.datadict.dictionary_VR(tag_address)
                    if tag_address in ds: del ds[tag_address]
                    ds.add_new(tag_address, vr, valor_a_escribir_clasificacion)
                
                elemento_escrito = ds.get(tag_keyword_clasificacion, None)
                valor_escrito_str = str(elemento_escrito.value) if elemento_escrito and hasattr(elemento_escrito, 'value') else \
                                  str(elemento_escrito) if not hasattr(elemento_escrito, 'value') and elemento_escrito is not None else \
                                  "(Tag no encontrado o valor None después de escribir!)"
                logger.info(f"[{sop_uid}] '{tag_keyword_clasificacion}' después de actualizar. Valor: '{valor_escrito_str}'")
            except KeyError:
                 logger.warning(f"Keyword '{tag_keyword_clasificacion}' no es un tag DICOM estándar. No se pudo determinar VR.")
            except Exception as e_set_tag:
                logger.error(f"[{sop_uid}] Error al establecer el tag '{tag_keyword_clasificacion}': {e_set_tag}", exc_info=True)
        else:
            logger.info(f"[{sop_uid}] No se escribió ningún valor de clasificación BAML en '{tag_keyword_clasificacion}'.")
        # --- FIN: Lógica para clasificación BAML ---
        
        # 3. Almacenar parámetros de linealización física (optional, separate from main linearization)
        if linearization_slope_param is not None and rqa_type_param is not None and private_creator_id_linealizacion is not None:
            logger.info(f"[{sop_uid}] Añadiendo parámetros de linealización física a cabecera: RQA={rqa_type_param}, Pendiente={linearization_slope_param:.4e}")
            ds = linealize.add_linearization_parameters_to_dicom(
                ds, rqa_type_param, linearization_slope_param, private_creator_id=private_creator_id_linealizacion )
        else:
            logger.debug(f"[{sop_uid}] No se proporcionaron todos los parámetros para linealización física.")

        # 4. Aplicar linealización con RescaleSlope/Intercept
        ds = _apply_rescale_tags_for_linearization(
            ds, pixel_values_calib, kerma_values_calib
        )
        logger.info(f"[{sop_uid}] Linealización por RescaleSlope/Intercept aplicada.")

        # SANEAMIENTO DE ImageType y SpecificCharacterSet
        logger.info(f"[{sop_uid}] Forzando ImageType a un valor conocido y válido.")
        try:
            image_type_tag = Tag(0x0008, 0x0008)
            if image_type_tag in ds:
                del ds[image_type_tag]
                logger.debug(f"[{sop_uid}] ImageType (0008,0008) original eliminado para recreación explícita.")
            valores_image_type_validos = ["DERIVED", "PRIMARY", "IMG_PROC_FINAL"] 
            ds.add_new(image_type_tag, "CS", valores_image_type_validos)
            logger.info(f"[{sop_uid}] ImageType (0008,0008) recreado con valor: {ds.ImageType}")
        except Exception as e_it_force:
            logger.error(f"[{sop_uid}] Error forzando ImageType: {e_it_force}", exc_info=True)

        if "SpecificCharacterSet" not in ds:
            ds.SpecificCharacterSet = "ISO_IR 192"
            logger.info(f"[{sop_uid}] SpecificCharacterSet no presente. Establecido a '{ds.SpecificCharacterSet}'.")
        elif ds.SpecificCharacterSet not in ["ISO_IR 192", "ISO 2022 IR 192"]:
             logger.warning(f"[{sop_uid}] SpecificCharacterSet actual es '{ds.SpecificCharacterSet}'. Considera cambiarlo a 'ISO_IR 192'.")
        
        # 5. Generar nuevo nombre de fichero
        img_num_fn = ds.get('InstanceNumber', ds.get('AcquisitionNumber', "X"))
        detector_id_fn = ds.get('DetectorID', 'NoDetID')
        kvp_fn = ds.get('KVP', 'NoKVP')
        try:
            exposure_uas_fn = float(ds.get('ExposureInuAs', 0.0))
        except ValueError:
            exposure_uas_fn = 0.0
            logger.warning(f"[{sop_uid}] Valor de ExposureInuAs no numérico: '{ds.get('ExposureInuAs', '')}'. Usando 0.0.")
        exposure_index_fn = ds.get('ExposureIndex', 'NoIE')
        new_filename_base = (
            f"Img{clean_filename_part(img_num_fn)}"
            f"_{clean_filename_part(detector_id_fn)}"
            f"_KVP{clean_filename_part(kvp_fn)}"
            f"_mAs{round(exposure_uas_fn / 1000.0, 2)}"
            f"_IE{clean_filename_part(exposure_index_fn)}" )
        max_len_base = 180 
        if len(new_filename_base) > max_len_base:
            new_filename_base = new_filename_base[:max_len_base]
            logger.warning(f"[{sop_uid}] Nombre de fichero base truncado a {max_len_base} caracteres.")
        new_filename = new_filename_base + ".dcm"
        output_filepath = output_base_dir / new_filename
        output_base_dir.mkdir(parents=True, exist_ok=True)

        # 6. Guardar el fichero DICOM procesado
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dcmwrite(str(output_filepath), ds)
        logger.info(f"[{sop_uid}] Archivo DICOM procesado y guardado como: {output_filepath}")
        return output_filepath
    except Exception as e:
        logger.error(f"Error fatal en process_and_prepare_dicom_for_pacs para {original_filename} (SOP UID: {sop_uid}): {e}", exc_info=True)
        return None

async def _test_pipeline_module():
    # This test function needs to be updated to reflect the new changes.
    # For now, it is left as is, but it will fail if run.
    pass
