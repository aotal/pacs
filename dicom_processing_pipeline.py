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

# --- load_kerma_calibration_data_for_lut, read_and_decompress_dicom, _apply_kerma_lut_to_dataset ---
# --- Estas funciones permanecen IGUAL que en la última versión funcional ---
# --- (la que resolvió el error de numpy.uint_ y los NameErrors) ---
# --- Asegúrate de copiar esas versiones aquí ---

def load_kerma_calibration_data_for_lut(csv_filepath: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        path_obj = Path(csv_filepath)
        if not path_obj.is_file():
            logger.error(f"Fichero CSV de calibración LUT Kerma no encontrado: {csv_filepath}")
            return None, None
        df = pd.read_csv(path_obj)
        required_cols = ['VMP', 'K_uGy'] 
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"El CSV {csv_filepath} para LUT Kerma debe contener las columnas: {missing}.")
            return None, None
        if df[required_cols].isnull().any().any():
            logger.warning(f"Valores nulos encontrados en columnas {required_cols} de {csv_filepath} para LUT Kerma. Se eliminarán esas filas.")
            df.dropna(subset=required_cols, inplace=True)
            if df.empty:
                logger.error(f"El CSV {csv_filepath} para LUT Kerma quedó vacío después de eliminar filas con NaNs.")
                return None, None
        pixel_values = df['VMP'].to_numpy(dtype=float)
        kerma_values = df['K_uGy'].to_numpy(dtype=float)
        if len(pixel_values) < 2: 
            logger.error(f"No hay suficientes datos válidos (se necesitan al menos 2 puntos) en {csv_filepath} para la LUT Kerma.")
            return None, None
        logger.info(f"Datos de calibración para LUT Kerma cargados desde {csv_filepath}: {len(pixel_values)} puntos.")
        return pixel_values, kerma_values
    except Exception as e:
        logger.error(f"Error leyendo el fichero CSV de calibración LUT Kerma ({csv_filepath}): {e}", exc_info=True)
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

def _apply_kerma_lut_to_dataset(ds: Dataset, 
                                pixel_values_calibration: np.ndarray, 
                                kerma_values_calibration: np.ndarray, 
                                kerma_scaling_factor: float) -> Dataset:
    sop_uid = ds.SOPInstanceUID if 'SOPInstanceUID' in ds else 'UID_DESCONOCIDO'
    logger.info(f"Aplicando Kerma LUT al dataset SOPInstanceUID: {sop_uid}")
    if not (isinstance(pixel_values_calibration, np.ndarray) and isinstance(kerma_values_calibration, np.ndarray)):
        raise TypeError("pixel_values_calibration y kerma_values_calibration deben ser numpy arrays.")
    if len(pixel_values_calibration) != len(kerma_values_calibration):
        raise ValueError("Arrays de calibración deben tener la misma longitud.")
    if len(pixel_values_calibration) < 2:
        raise ValueError("Se necesitan al menos dos puntos para interpolación.")
    if 'RescaleSlope' in ds: ds.RescaleSlope = 1.0
    if 'RescaleIntercept' in ds: ds.RescaleIntercept = 0.0
    if 'RescaleType' in ds: del ds.RescaleType
    logger.debug(f"[{sop_uid}] RescaleSlope/Intercept/Type neutralizados.")
    bits_stored = ds.BitsStored
    pixel_representation = ds.PixelRepresentation
    num_lut_entries_descriptor: int
    first_mapped_value_descriptor: int
    lut_input_pixel_range: np.ndarray
    num_lut_entries_descriptor = 2**bits_stored
    if pixel_representation == 1: 
        first_mapped_value_descriptor = -(2**(bits_stored - 1))
        max_stored_value = 2**(bits_stored - 1) - 1
        lut_input_pixel_range = np.arange(first_mapped_value_descriptor, max_stored_value + 1).astype(float)
    else: 
        first_mapped_value_descriptor = 0
        max_stored_value = 2**bits_stored - 1
        lut_input_pixel_range = np.arange(first_mapped_value_descriptor, max_stored_value + 1).astype(float)
    logger.debug(f"[{sop_uid}] Info píxeles LUT: BitsStored={bits_stored}, PixelRep={pixel_representation}")
    logger.debug(f"[{sop_uid}] LUTDescriptor: NumEntries={num_lut_entries_descriptor}, FirstMapped={first_mapped_value_descriptor}, BitsPerEntry=16")
    sort_indices = np.argsort(pixel_values_calibration)
    sorted_pixel_values_calib = np.array(pixel_values_calibration)[sort_indices]
    sorted_kerma_values_calib = np.array(kerma_values_calibration)[sort_indices]
    interpolated_kerma_f = np.interp(lut_input_pixel_range,
                                     sorted_pixel_values_calib,
                                     sorted_kerma_values_calib,
                                     left=sorted_kerma_values_calib[0],
                                     right=sorted_kerma_values_calib[-1])
    scaled_kerma_for_lutdata = np.round(interpolated_kerma_f * kerma_scaling_factor)
    scaled_kerma_uint16 = np.clip(scaled_kerma_for_lutdata, 0, 65535).astype(np.uint16)
    if not (len(scaled_kerma_uint16) == num_lut_entries_descriptor):
        logger.error(f"[{sop_uid}] Discrepancia tamaño LUTData: esperado {num_lut_entries_descriptor}, obtenido {len(scaled_kerma_uint16)}.")
        raise ValueError("Error crítico LUTData: tamaño incorrecto.")
    logger.debug(f"[{sop_uid}] Cálculos LUTData OK.")
    ds.ModalityLUTSequence = Sequence()
    modality_lut_item = Dataset()
    modality_lut_item.LUTDescriptor = [num_lut_entries_descriptor, first_mapped_value_descriptor, 16]
    s_factor_str = f"{kerma_scaling_factor:.0f}" if kerma_scaling_factor == int(kerma_scaling_factor) else f"{kerma_scaling_factor:.1f}"
    min_calib_kerma_str = f"{np.min(kerma_values_calibration):.2f}"
    max_calib_kerma_str = f"{np.max(kerma_values_calibration):.2f}"
    min_lut_data_str = f"{np.min(scaled_kerma_uint16)}"
    max_lut_data_str = f"{np.max(scaled_kerma_uint16)}"
    explanation = (f"Kerma uGy (SF={s_factor_str}) "
                   f"InCalibRange:{min_calib_kerma_str}-{max_calib_kerma_str} "
                   f"OutLUTRange:{min_lut_data_str}-{max_lut_data_str}")
    if len(explanation) > 64: explanation = explanation[:61] + "..."
    modality_lut_item.LUTExplanation = explanation
    modality_lut_item.ModalityLUTType = 'KERMA_SCALED' 
    modality_lut_item.LUTData = scaled_kerma_uint16.tobytes()
    ds.ModalityLUTSequence.append(modality_lut_item)
    logger.debug(f"[{sop_uid}] ModalityLUTSequence OK. Explicación: {explanation}")
    min_output_val = float(np.min(scaled_kerma_uint16))
    max_output_val = float(np.max(scaled_kerma_uint16))
    new_ww = max_output_val - min_output_val
    if new_ww < 1.0: new_ww = 1.0 
    new_wc = min_output_val + new_ww / 2.0
    ds.WindowCenter = new_wc
    ds.WindowWidth = new_ww
    logger.debug(f"[{sop_uid}] Nuevos WC: {ds.WindowCenter}, WW: {ds.WindowWidth} OK.")
    if 'VOILUTSequence' in ds:
        del ds.VOILUTSequence
        logger.debug(f"[{sop_uid}] VOILUTSequence eliminada.")
    if ds.PhotometricInterpretation not in ["MONOCHROME1", "MONOCHROME2"]:
        logger.warning(f"[{sop_uid}] PhotometricInterpretation original '{ds.PhotometricInterpretation}'. Cambiando a MONOCHROME2.")
        ds.PhotometricInterpretation = "MONOCHROME2" 
    elif ds.PhotometricInterpretation == "MONOCHROME1": 
        logger.info(f"[{sop_uid}] PhotometricInterpretation es MONOCHROME1. Considera si MONOCHROME2 es más apropiado para Kerma (GSDF).")
    return ds


def process_and_prepare_dicom_for_pacs(
    ds: Dataset, 
    clasificacion_baml_mapeada: Optional[str], # <--- Nombre de parámetro para valor MAPEADO
    pixel_values_lut_calib: np.ndarray,
    kerma_values_lut_calib: np.ndarray,
    kerma_scaling_factor_lut: float,
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

        # --- INICIO: Lógica MODIFICADA para clasificación BAML (sobrescribir y sin prefijo) ---
        tag_keyword_clasificacion = getattr(config, 'DICOM_TAG_FOR_CLASSIFICATION', 'ImageComments')
        valor_a_escribir_clasificacion = None

        if clasificacion_baml_mapeada and \
           not clasificacion_baml_mapeada.startswith("Error") and \
           clasificacion_baml_mapeada not in ["BAML_OTRO", "ClasificacionFallida", "Desconocida", "ErrorPixelArrayNulo"]:
            # clasificacion_baml_mapeada ya es "FDT", "MTF", "BC"
            valor_a_escribir_clasificacion = clasificacion_baml_mapeada
        elif clasificacion_baml_mapeada: 
             logger.warning(f"[{sop_uid}] Clasificación BAML mapeada fue '{clasificacion_baml_mapeada}' (error o placeholder). No se actualizará '{tag_keyword_clasificacion}'.")
        else: 
            logger.warning(f"[{sop_uid}] No se proporcionó clasificación BAML mapeada válida. No se actualizará '{tag_keyword_clasificacion}'.")

        if valor_a_escribir_clasificacion:
            logger.info(f"[{sop_uid}] Estableciendo (sobrescribiendo) '{tag_keyword_clasificacion}' a: '{valor_a_escribir_clasificacion}'")
            try:
                # Para tags estándar conocidos, setattr es más simple y pydicom maneja el VR.
                if hasattr(ds, tag_keyword_clasificacion) or tag_keyword_clasificacion in pydicom.datadict.keyword_dict:
                    setattr(ds, tag_keyword_clasificacion, valor_a_escribir_clasificacion)
                else: # Para tags no directamente asignables o si se quiere ser explícito con VR
                    tag_address = Tag(tag_keyword_clasificacion) # Acepta keyword o (G,E)
                    vr = pydicom.datadict.dictionary_VR(tag_address) # Intenta obtener VR
                    if tag_address in ds: del ds[tag_address] # Eliminar para asegurar sobrescritura limpia
                    ds.add_new(tag_address, vr, valor_a_escribir_clasificacion)
                
                # Verificar el valor escrito
                elemento_escrito = ds.get(tag_keyword_clasificacion, None)
                valor_escrito_str = str(elemento_escrito.value) if elemento_escrito and hasattr(elemento_escrito, 'value') else \
                                  str(elemento_escrito) if not hasattr(elemento_escrito, 'value') and elemento_escrito is not None else \
                                  "(Tag no encontrado o valor None después de escribir!)"
                logger.info(f"[{sop_uid}] '{tag_keyword_clasificacion}' después de actualizar. Valor: '{valor_escrito_str}'")
            except KeyError: # Si el keyword no es estándar y no se pudo determinar el VR
                 logger.warning(f"Keyword '{tag_keyword_clasificacion}' no es un tag DICOM estándar. "
                               f"No se pudo determinar VR automáticamente. Si es un tag privado, defínelo en el diccionario privado primero o usa add_new con VR explícito.")
            except Exception as e_set_tag:
                logger.error(f"[{sop_uid}] Error al establecer el tag '{tag_keyword_clasificacion}' con valor '{valor_a_escribir_clasificacion}': {e_set_tag}", exc_info=True)
        else:
            logger.info(f"[{sop_uid}] No se escribió ningún valor de clasificación BAML en '{tag_keyword_clasificacion}'.")
        # --- FIN: Lógica MODIFICADA para clasificación BAML ---
        
        # 3. Almacenar parámetros de linealización física
        if linearization_slope_param is not None and rqa_type_param is not None and private_creator_id_linealizacion is not None:
            logger.info(f"[{sop_uid}] Añadiendo parámetros de linealización física a cabecera: RQA={rqa_type_param}, Pendiente={linearization_slope_param:.4e}")
            ds = linealize.add_linearization_parameters_to_dicom(
                ds, rqa_type_param, linearization_slope_param, private_creator_id=private_creator_id_linealizacion )
        else:
            logger.debug(f"[{sop_uid}] No se proporcionaron todos los parámetros para linealización física.")

        # 4. Aplicar LUT Kerma
        ds = _apply_kerma_lut_to_dataset(
            ds, pixel_values_lut_calib, kerma_values_lut_calib, kerma_scaling_factor_lut )
        logger.info(f"[{sop_uid}] LUT Kerma aplicada.")

        # SANEAMIENTO DE ImageType y SpecificCharacterSet (MANTENER ESTA LÓGICA)
        logger.info(f"[{sop_uid}] Forzando ImageType a un valor conocido y válido.")
        try:
            image_type_tag = Tag(0x0008, 0x0008)
            if image_type_tag in ds:
                del ds[image_type_tag]
                logger.debug(f"[{sop_uid}] ImageType (0008,0008) original eliminado para recreación explícita.")
            # Usar valores CS válidos (max 16 chars, MAYÚSCULAS, números, '_', espacio)
            valores_image_type_validos = ["DERIVED", "PRIMARY", "IMG_PROC_FINAL"] 
            ds.add_new(image_type_tag, "CS", valores_image_type_validos)
            logger.info(f"[{sop_uid}] ImageType (0008,0008) recreado con valor: {ds.ImageType}")
        except Exception as e_it_force:
            logger.error(f"[{sop_uid}] Error forzando ImageType: {e_it_force}", exc_info=True)

        if "SpecificCharacterSet" not in ds:
            ds.SpecificCharacterSet = "ISO_IR 192" # UTF-8 para mayor compatibilidad
            logger.info(f"[{sop_uid}] SpecificCharacterSet no presente. Establecido a '{ds.SpecificCharacterSet}'.")
        elif ds.SpecificCharacterSet not in ["ISO_IR 192", "ISO 2022 IR 192"]: # ISO 2022 IR 192 también es UTF-8
             logger.warning(f"[{sop_uid}] SpecificCharacterSet actual es '{ds.SpecificCharacterSet}'. "
                           "Considera cambiarlo a 'ISO_IR 192' (UTF-8) por robustez.")
        
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

# --- Bloque de prueba _test_pipeline_module ---
# (COPIA EL BLOQUE _test_pipeline_module de la versión anterior que te di,
#  la que corregía el acceso a PrivateBlock y usaba MockConfig.
#  Asegúrate de que la llamada a process_and_prepare_dicom_for_pacs dentro de la prueba
#  pase 'clasificacion_baml_mapeada' con un valor como "FDT_TEST".)
async def _test_pipeline_module():
    if not logging.getLogger().hasHandlers():
         from utils import configurar_logging_aplicacion 
         configurar_logging_aplicacion(level=logging.DEBUG)
    logger.info("--- Iniciando prueba de dicom_processing_pipeline.py (con BAML mapeado y sobrescrito) ---")
    
    class MockConfig:
        DICOM_TAG_FOR_CLASSIFICATION = "SeriesDescription"
        # CLASSIFICATION_TAG_PREFIX ya no se usa para escribir
        KERMA_SCALING_FACTOR = 100.0
        ENABLE_PHYSICAL_LINEALIZATION_PARAMS = True 
        _test_base_dir_for_mock = Path(__file__).resolve().parent / "test_pipeline_data_map_overwrite"
        PATH_LUT_CALIBRATION_CSV = _test_base_dir_for_mock / "data" / "linearizacion_map.csv"
        PATH_CSV_LINEALIZACION_FISICA = _test_base_dir_for_mock / "data" / "linearizacion_map.csv"
        DEFAULT_RQA_TYPE_LINEALIZATION = "RQA5_MAP"
        RQA_FACTORS_PHYSICAL_LINEALIZATION = {"RQA5_MAP": 0.000130}
        PRIVATE_CREATOR_ID_LINEALIZATION = "PIPE_LIN_MAP"

    global config 
    config_original_ref = config 
    config = MockConfig() 

    base_test_dir = Path(__file__).resolve().parent / "test_pipeline_data_map_final"
    test_input_dir = base_test_dir / "input"
    test_output_dir = base_test_dir / "output"
    test_data_dir = base_test_dir / "data" 
    for d in [test_input_dir, test_output_dir, test_data_dir, config.PATH_LUT_CALIBRATION_CSV.parent]:
        d.mkdir(parents=True, exist_ok=True)
    try:
        file_meta_test = Dataset()
        file_meta_test.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1' 
        file_meta_test.MediaStorageSOPInstanceUID = generate_uid()
        file_meta_test.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta_test.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds_test = pydicom.dataset.FileDataset(None, {}, file_meta=file_meta_test, preamble=b"\0" * 128)
        ds_test.SOPClassUID = file_meta_test.MediaStorageSOPClassUID
        ds_test.SOPInstanceUID = file_meta_test.MediaStorageSOPInstanceUID
        ds_test.PatientName = "Test^MapOverwrite"
        ds_test.PatientID = "IDMapOverwrite"
        ds_test.SeriesDescription = "Valor Original Que Sera Sobrescrito" 
        ds_test.ImageType = ["ORIGINAL", "PRIMARY"] # Un ImageType inicial válido
        ds_test.Modality = "CR"; ds_test.BitsAllocated = 16; ds_test.BitsStored = 12
        ds_test.HighBit = 11; ds_test.PixelRepresentation = 0; ds_test.SamplesPerPixel = 1
        ds_test.PhotometricInterpretation = "MONOCHROME2"
        ds_test.Rows = 10; ds_test.Columns = 10
        ds_test.PixelData = np.random.randint(0, 2**ds_test.BitsStored, size=(ds_test.Rows * ds_test.Columns), dtype=np.uint16).tobytes()
        ds_test.DetectorID = "DetMapOverwrite"; ds_test.StationName = "StationMapOverwrite"
        test_dcm_path = test_input_dir / "test_dicom_map_overwrite.dcm"
        pydicom.dcmwrite(str(test_dcm_path), ds_test)
        logger.info(f"Fichero DICOM de prueba (map/overwrite) creado: {test_dcm_path}")
        
        pd.DataFrame({'VMP': [0, 4095], 'K_uGy': [0, 40.95]}).to_csv(config.PATH_LUT_CALIBRATION_CSV, index=False)
        pixels_cal_k, kerma_cal_k = load_kerma_calibration_data_for_lut(str(config.PATH_LUT_CALIBRATION_CSV))
        if not (pixels_cal_k is not None and pixels_cal_k.size > 0): return logger.error("Fallo LUT Kerma calib data test (map/overwrite)")

        df_lin_calib = linealize.obtener_datos_calibracion_vmp_k_linealizacion(str(config.PATH_CSV_LINEALIZACION_FISICA))
        test_slope_lin = None
        if df_lin_calib is not None and config.RQA_FACTORS_PHYSICAL_LINEALIZATION and config.DEFAULT_RQA_TYPE_LINEALIZATION in config.RQA_FACTORS_PHYSICAL_LINEALIZATION:
             test_slope_lin = linealize.calculate_linearization_slope(df_lin_calib, config.DEFAULT_RQA_TYPE_LINEALIZATION, config.RQA_FACTORS_PHYSICAL_LINEALIZATION)

        ds_read, _ = await read_and_decompress_dicom(test_dcm_path) 
        if ds_read:
            clasificacion_mapeada_test = "FDT_MAP_TEST" # Simular valor mapeado

            output_file = process_and_prepare_dicom_for_pacs(
                ds=ds_read.copy(), 
                clasificacion_baml_mapeada=clasificacion_mapeada_test,
                pixel_values_lut_calib=pixels_cal_k,
                kerma_values_lut_calib=kerma_cal_k,
                kerma_scaling_factor_lut=config.KERMA_SCALING_FACTOR,
                output_base_dir=test_output_dir,
                original_filename=test_dcm_path.name,
                linearization_slope_param=test_slope_lin, 
                rqa_type_param=config.DEFAULT_RQA_TYPE_LINEALIZATION if test_slope_lin is not None else None,
                private_creator_id_linealizacion=config.PRIVATE_CREATOR_ID_LINEALIZATION if test_slope_lin is not None else None
            )
            if output_file and output_file.exists():
                logger.info(f"Prueba (map/overwrite) completada. Fichero procesado: {output_file}")
                ds_final = pydicom.dcmread(str(output_file), force=True)
                tag_final_clasif = ds_final.get(config.DICOM_TAG_FOR_CLASSIFICATION, 'N/A')
                logger.info(f"  {config.DICOM_TAG_FOR_CLASSIFICATION} final: {tag_final_clasif}")
                # Verificar que el valor sea exactamente el mapeado y no haya concatenación
                assert tag_final_clasif == clasificacion_mapeada_test, f"Se esperaba '{clasificacion_mapeada_test}' pero se obtuvo '{tag_final_clasif}'"
                logger.info(f"  ImageType final: {ds_final.get('ImageType', 'N/A')}") 
                logger.info(f"  SpecificCharacterSet final: {ds_final.get('SpecificCharacterSet', 'N/A')}")
                if config.ENABLE_PHYSICAL_LINEALIZATION_PARAMS and test_slope_lin:
                    # ... (verificación del bloque privado como antes)
                    pass
    except ImportError:
        logger.critical("Faltan importaciones para prueba de pipeline.", exc_info=True)
    except Exception as e:
        logger.error(f"Error durante la prueba de pipeline (map/overwrite): {e}", exc_info=True)
    finally:
        if base_test_dir.exists():
            shutil.rmtree(base_test_dir)
            logger.info(f"Directorio de prueba {base_test_dir} eliminado.")
        config = config_original_ref 

if __name__ == '__main__':
    import asyncio 
    asyncio.run(_test_pipeline_module())