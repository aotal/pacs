# linealize.py
import logging
import warnings
from typing import Dict, Optional, Tuple # Union también podría ser útil

import numpy as np
import pandas as pd
import pydicom # Para añadir tags al dataset DICOM
from pydicom.dataset import Dataset # Para type hinting explícito
from pydicom.uid import generate_uid # Para SOPInstanceUID de prueba

logger = logging.getLogger(__name__)

# --- Constantes (si son específicas de este módulo) ---
# Estos factores RQA deberían venir de config.py o ser pasados como argumento
RQA_FACTORS_EXAMPLE: Dict[str, float] = {
    "RQA5": 0.000123, # Reemplaza con tus valores reales (SNR_in^2 / 1000)
    "RQA9": 0.000456,
    # ...otros RQA...
}
EPSILON = 1e-9 # Para evitar divisiones por cero

# --- Funciones de Carga de Datos de Calibración para Linealización Física ---

def obtener_datos_calibracion_vmp_k_linealizacion(
    ruta_archivo_csv: str,
) -> Optional[pd.DataFrame]:
    """
    Carga un archivo CSV que contiene los datos de calibración VMP vs Kerma
    usados para calcular la pendiente de linealización física.
    Se espera que el CSV tenga columnas como 'K_uGy' y 'VMP'.
    """
    try:
        # Usar pathlib para manejo de rutas es más robusto
        from pathlib import Path
        path_obj = Path(ruta_archivo_csv)
        if not path_obj.is_file():
            logger.error(f"Fichero CSV de calibración (para linealización física) no encontrado: {ruta_archivo_csv}")
            return None

        df = pd.read_csv(path_obj)
        logger.info(f"Datos de calibración VMP vs K (para linealización física) cargados desde: {ruta_archivo_csv}")
        
        required_columns = ['K_uGy', 'VMP']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Columnas requeridas {missing} no encontradas en {ruta_archivo_csv} para linealización.")
            return None
        
        # Validar que haya datos después de eliminar NaNs en columnas críticas
        if df[required_columns].isnull().any().any():
            logger.warning(f"Valores nulos encontrados en columnas {required_columns} de {ruta_archivo_csv}. Se eliminarán esas filas.")
            df.dropna(subset=required_columns, inplace=True)
            if df.empty:
                logger.error(f"El CSV {ruta_archivo_csv} quedó vacío después de eliminar NaNs en {required_columns}.")
                return None
        
        if len(df) < 2: # Necesita al menos dos puntos para la pendiente
             logger.error(f"No hay suficientes datos válidos (se necesitan al menos 2 puntos) en {ruta_archivo_csv} para calcular la pendiente.")
             return None

        return df
    except FileNotFoundError: # Ya cubierto por Path.is_file(), pero por si acaso
        logger.error(f"Fichero CSV de calibración (para linealización física) no encontrado: {ruta_archivo_csv}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Fichero CSV de calibración (para linealización física) {ruta_archivo_csv} está vacío.")
        return None
    except Exception as e:
        logger.exception(f"Error al leer datos de calibración (para linealización física) desde {ruta_archivo_csv}: {e}")
        return None


# --- Funciones de Cálculo de Parámetros de Linealización ---

def calculate_linearization_slope(
    calibration_df: pd.DataFrame, 
    rqa_type: str, 
    rqa_factors_dict: Dict[str, float] # = RQA_FACTORS_EXAMPLE # Es mejor pasar esto explícitamente
) -> Optional[float]:
    """
    Calcula la pendiente (VMP vs quanta/area) para un RQA dado.
    """
    try:
        if not isinstance(calibration_df, pd.DataFrame): raise TypeError("calibration_df debe ser DataFrame.")
        if not isinstance(rqa_factors_dict, dict): raise TypeError("rqa_factors_dict debe ser dict.")
        if rqa_type not in rqa_factors_dict: 
            logger.error(f"RQA type '{rqa_type}' no encontrado en rqa_factors_dict. Disponibles: {list(rqa_factors_dict.keys())}")
            raise ValueError(f"RQA type '{rqa_type}' no en rqa_factors_dict.")
        if not all(col in calibration_df.columns for col in ['K_uGy', 'VMP']): 
            raise ValueError("El DataFrame de calibración debe contener las columnas 'K_uGy' y 'VMP'.")

        factor_lin = rqa_factors_dict[rqa_type] 
        snr_in_squared_factor = factor_lin * 1000.0
        
        valid_cal_data = calibration_df[
            (calibration_df['K_uGy'] > EPSILON) & 
            (np.isfinite(calibration_df['VMP'])) & # VMP debe ser finito
            (np.isfinite(calibration_df['K_uGy']))  # K_uGy también debe ser finito
        ].copy()

        if valid_cal_data.empty: 
            logger.warning(f"No hay puntos de calibración válidos (K_uGy > {EPSILON} y VMP/K_uGy finitos) para {rqa_type}.")
            return None

        valid_cal_data['quanta_per_area'] = valid_cal_data['K_uGy'] * snr_in_squared_factor
        
        x_values = valid_cal_data['quanta_per_area'].values
        y_values = valid_cal_data['VMP'].values
        
        valid_points_mask = (np.abs(x_values) > EPSILON) & np.isfinite(x_values) & np.isfinite(y_values)
        x_masked = x_values[valid_points_mask]
        y_masked = y_values[valid_points_mask]

        if len(x_masked) < 2: # Necesita al menos dos puntos para un ajuste lineal robusto
            logger.warning(f"No quedan suficientes puntos válidos ({len(x_masked)}) para el cálculo de la pendiente para {rqa_type} después de filtrar.")
            return None
        
        # Estimación de la pendiente: sum(x*y) / sum(x*x) para el modelo y = slope * x (pasa por el origen)
        slope_prime = np.sum(x_masked * y_masked) / np.sum(x_masked**2)

        if abs(slope_prime) < EPSILON:
            logger.warning(f"Pendiente de linealización calculada para {rqa_type} ({slope_prime:.2e}) es demasiado cercana a cero.")
            return None 
            
        logger.info(f"Pendiente de linealización calculada para {rqa_type}: {slope_prime:.6e}")
        return float(slope_prime)
    except Exception as e:
        logger.warning(f"No se pudo calcular la pendiente de linealización para {rqa_type}: {e}", exc_info=True)
        return None


# --- Funciones para Aplicar Linealización (a un array de píxeles, si se necesita fuera del PACS) ---

def linearize_pixel_array(
    pixel_array: np.ndarray, 
    linearization_slope: float
) -> Optional[np.ndarray]:
    """
    Linealiza un array de píxeles (ya preprocesado) dividiéndolo por la pendiente de linealización.
    """
    if not isinstance(pixel_array, np.ndarray):
        logger.error("Entrada 'pixel_array' debe ser un numpy array.")
        return None
    if not isinstance(linearization_slope, (float, np.floating)) or abs(linearization_slope) < EPSILON:
        logger.error(f"Pendiente de linealización inválida o cercana a cero: {linearization_slope}")
        return None
        
    try:
        image_float = pixel_array.astype(np.float64)
        linearized_image = image_float / linearization_slope
        logger.info("Array de píxeles linealizado (división por pendiente).")
        return linearized_image
    except Exception as e: # Captura genérica, ZeroDivisionError ya cubierto por el check de slope
        logger.exception(f"Error inesperado durante la linealización del array de píxeles: {e}")
        return None


# --- Funciones de Ayuda para VMP (si se usan para derivar la pendiente) ---

def calculate_vmp_roi(imagen: np.ndarray, halfroi: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Calcula el Valor Medio de Píxel (VMP) y la desviación estándar en una ROI cuadrada central.
    """
    try:
        if not isinstance(imagen, np.ndarray) or imagen.ndim != 2:
            logger.warning(f"Imagen para VMP no es 2D (dimensiones: {imagen.ndim}). Se devuelve None.")
            return None, None
        if halfroi <= 0:
            logger.warning(f"Tamaño de halfroi ({halfroi}) debe ser positivo. Se devuelve None.")
            return None, None
            
        img_h, img_w = imagen.shape
        centro_y, centro_x = img_h // 2, img_w // 2
        
        y_start, y_end = max(0, centro_y - halfroi), min(img_h, centro_y + halfroi)
        x_start, x_end = max(0, centro_x - halfroi), min(img_w, centro_x + halfroi)

        if y_start >= y_end or x_start >= x_end:
            logger.warning(f"ROI para VMP tiene tamaño cero o inválido: y[{y_start}:{y_end}], x[{x_start}:{x_end}]")
            return None, None
            
        roi = imagen[y_start:y_end, x_start:x_end]
        if roi.size == 0:
            logger.warning("ROI para VMP está vacía después del slicing.")
            return None, None

        vmp = np.mean(roi)
        std = np.std(roi)
        logger.debug(f"VMP ROI calculado: {vmp:.2f}, StdDev: {std:.2f}")
        return float(vmp), float(std)
    except Exception as e:
        logger.exception(f"Error al calcular VMP en ROI: {e}")
        return None, None

# --- Funciones para almacenar parámetros de linealización en DICOM ---

def add_linearization_parameters_to_dicom(
    ds: Dataset, 
    rqa_type: str, 
    linearization_slope: float,
    private_creator_id: str = "LINEALIZATION_PARAMS_RFB" # RFB: Request For Behavior - elige un ID único
) -> Dataset:
    """
    Añade los parámetros de linealización calculados a la cabecera DICOM
    usando un bloque privado. No modifica los datos de píxeles.
    """
    try:
        # Grupo privado (impar) para los parámetros de linealización.
        # Elige un grupo que sepas que está libre en tus sistemas. 0x00F1 es solo un ejemplo.
        # El estándar reserva (gggg,00xx) para Private Creator Data Elements.
        # Los elementos de datos privados dentro del bloque son (gggg,xxee) donde xx es el bloque (10-FF).
        private_group = 0x00F1 
        
        # Obtener o crear el bloque privado
        # El private_creator_id identifica tu bloque de datos privados.
        block = ds.private_block(private_group, private_creator_id, create=True)
        
        # Definir los offsets de los elementos dentro del bloque privado (elementos 10-FF)
        # (0x00F1, private_creator_id_offset) -> almacena 'private_creator_id'
        # (0x00F1, 0xXX10) -> RQA Type
        # (0x00F1, 0xXX11) -> Linearization Slope
        # El 'XX' es el offset que Pydicom asigna o encuentra para tu private_creator_id.
        # No necesitas saberlo explícitamente para añadir datos, solo el offset del elemento (ej. 0x10, 0x11).
        
        block.add_new(0x10, "LO", rqa_type) # Elemento offset 0x10: RQA Type
        block.add_new(0x11, "DS", f"{linearization_slope:.8e}") # Elemento offset 0x11: Linearization Slope
        
        logger.info(f"Parámetros de linealización (RQA={rqa_type}, Pendiente={linearization_slope:.4e}) "
                    f"añadidos al dataset DICOM en bloque privado (Grupo:0x{private_group:04X}, Creador:'{private_creator_id}').")
        
    except Exception as e:
        logger.exception(f"Error añadiendo parámetros de linealización al dataset DICOM: {e}")
        # No relanzar para no detener el flujo, pero el ds no tendrá los tags.
        
    return ds


if __name__ == '__main__':
    from pathlib import Path # Para pruebas
    import shutil # Para limpiar directorio de prueba

    # Configurar logging básico para las pruebas de este módulo
    if not logging.getLogger().hasHandlers(): # Evitar añadir handlers múltiples si se importa
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Pruebas para linealize.py ---")

    # Crear un DataFrame de calibración de ejemplo
    sample_cal_data_dict = {
        'K_uGy': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 0.05, np.nan, 15.0], 
        'VMP':   [10,  50,  100, 200, 500, 1000, 5,    1200,   1500]  
    }
    test_csv_dir = Path("temp_linealize_test_data")
    test_csv_dir.mkdir(exist_ok=True)
    test_csv_path = test_csv_dir / "test_linearizacion_fisica.csv"
    
    temp_df_for_csv = pd.DataFrame(sample_cal_data_dict)
    temp_df_for_csv.to_csv(test_csv_path, index=False)
    logger.info(f"CSV de prueba creado en: {test_csv_path}")

    cal_df = obtener_datos_calibracion_vmp_k_linealizacion(str(test_csv_path))
    if cal_df is not None:
        print(f"\nDataFrame de calibración cargado (después de limpiar NaNs en K_uGy/VMP si aplica):\n{cal_df}")

        rqa = "RQA5" 
        slope = calculate_linearization_slope(cal_df, rqa, RQA_FACTORS_EXAMPLE)
        if slope:
            print(f"\nPendiente de linealización calculada para {rqa}: {slope:.6e}")

            test_pixel_array = np.array([[50, 100], [200, 500]], dtype=np.float32)
            linearized_arr = linearize_pixel_array(test_pixel_array, slope)
            if linearized_arr is not None:
                print(f"Array original (VMP):\n{test_pixel_array}")
                print(f"Array linealizado (quanta/area aproximado):\n{linearized_arr}")

            ds_test = Dataset()
            ds_test.PatientName = "Test^LinearizeParams"
            ds_test.SOPInstanceUID = generate_uid() 
            ds_test.file_meta = Dataset() 
            ds_test.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

            # Definir el private_creator_id que se usó en la función para la prueba
            test_private_creator_id = "MY_LIN_PARAMS"
            ds_modified = add_linearization_parameters_to_dicom(ds_test, rqa, slope, test_private_creator_id)
            print("\nDataset DICOM con parámetros de linealización:")
            
            found_block = False
            # Definir private_group aquí para que esté en el ámbito de esta prueba
            private_group_test = 0x00F1 # El mismo grupo usado en la función

            for i in range(0x10, 0xFF): 
                creator_tag = (private_group_test, i) # CORRECCIÓN: Usar private_group_test
                if creator_tag in ds_modified and ds_modified[creator_tag].value == test_private_creator_id:
                    print(f"  Bloque privado encontrado con creador '{ds_modified[creator_tag].value}' en offset 0x{i:02X}")
                    # Acceder a los elementos usando el offset del bloque 'i'
                    # El elemento es (grupo, (offset_bloque << 8) + offset_elemento_en_bloque)
                    # Pydicom < 2.0: (grupo, offset_bloque*0x100 + offset_elemento_en_bloque)
                    # Pydicom >= 2.0: ds.private_block(group, creator_id).get(element_offset) es más fácil
                    # Para la verificación manual de tags como en la prueba:
                    rqa_tag_in_block = (private_group_test, (i << 8) | 0x10) 
                    slope_tag_in_block = (private_group_test, (i << 8) | 0x11)
                    
                    if rqa_tag_in_block in ds_modified:
                         print(f"    RQA Type ({rqa_tag_in_block}): {ds_modified[rqa_tag_in_block].value}")
                    else:
                         print(f"    Tag RQA Type ({rqa_tag_in_block}) no encontrado en el bloque.")
                    if slope_tag_in_block in ds_modified:
                         print(f"    Slope ({slope_tag_in_block}): {ds_modified[slope_tag_in_block].value}")
                    else:
                         print(f"    Tag Slope ({slope_tag_in_block}) no encontrado en el bloque.")
                    found_block = True
                    break
            if not found_block:
                print(f"  No se encontró el bloque privado '{test_private_creator_id}' como se esperaba.")
        else:
            print(f"\nNo se pudo calcular la pendiente para {rqa}.")
    else:
        print("\nNo se pudieron cargar los datos de calibración desde el CSV de prueba.")


    img_test_vmp = np.array([[float(i+j) for i in range(10)] for j in range(0,100,10)], dtype=np.float32)
    vmp, std = calculate_vmp_roi(img_test_vmp, 2)
    if vmp is not None:
        print(f"\nPrueba VMP: Media={vmp:.2f}, StdDev={std:.2f}")
    
    vmp_zero_roi, _ = calculate_vmp_roi(img_test_vmp, 0)
    if vmp_zero_roi is None:
        print("Prueba VMP con halfroi=0 devolvió None como se esperaba.")

    vmp_small_img, _ = calculate_vmp_roi(np.array([[1,2],[3,4]], dtype=float), 5) 
    if vmp_small_img is not None:
         print(f"Prueba VMP con halfroi > tamaño imagen: Media={vmp_small_img:.2f}")

    if test_csv_dir.exists():
        shutil.rmtree(test_csv_dir)
        logger.info(f"Directorio de prueba {test_csv_dir} eliminado.")