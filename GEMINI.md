

# gemini.md

## 1\. Resumen del Proyecto

Este proyecto es un pipeline de software en Python diseñado para procesar ficheros de imagen médica en formato DICOM de manera automatizada. El sistema realiza una serie de tareas que incluyen la lectura y descompresión de imágenes, la modificación de metadatos (tags DICOM), la calibración de la dosis de radiación, la clasificación de imágenes mediante un modelo de IA (a través de BAML), y finalmente, el envío de los ficheros procesados a un servidor de almacenamiento de imágenes médicas (PACS).

El sistema es modular y asíncrono, lo que le permite manejar operaciones de red y de entrada/salida de manera eficiente. La configuración se centraliza en un archivo `config.py`, y utiliza variables de entorno para gestionar credenciales sensibles.

## 2\. Flujo de Trabajo Detallado

El script `main.py` actúa como el orquestador principal, ejecutando la siguiente secuencia de pasos para cada fichero DICOM de entrada:

1.  **Inicio y Carga de Configuración**: Se carga la configuración de logging, las rutas de los directorios de entrada/salida y los datos de calibración desde los ficheros CSV especificados en `config.py`. Hay dos tipos de datos de calibración:

      * **LUT Kerma**: Para linealizar los valores de los píxeles y representarlos en unidades de Kerma (µGy). Es obligatorio.
      * **Linealización Física**: Para calcular parámetros físicos (una pendiente de linealización) que se guardan en la cabecera, pero sin alterar los píxeles. Esta funcionalidad es opcional y se activa con `ENABLE_PHYSICAL_LINEALIZATION_PARAMS` en la configuración.

2.  **Lectura y Descompresión del DICOM** (`dicom_processing_pipeline.read_and_decompress_dicom`):

      * El fichero DICOM se lee desde el disco.
      * Si el fichero está comprimido (ej. JPEG Lossless), se descomprime en memoria para acceder a los datos de los píxeles.
      * Se extraen el dataset de pydicom (`ds`) y el array de píxeles (`pixel_array`).

3.  **Clasificación de Imagen con BAML** (`baml_classification.obtener_clasificacion_baml`):

      * El `pixel_array` se convierte a una imagen PNG y se codifica en base64.
      * Esta cadena base64 se envía a un servicio de IA a través de un cliente BAML.
      * El servicio devuelve una clasificación (ej. "Type1"), que es mapeada a un código interno definido por el sistema (ej. "FDT").

4.  **Procesamiento y Preparación del DICOM para PACS** (`dicom_processing_pipeline.process_and_prepare_dicom_for_pacs`): Este es el paso central donde se aplican la mayoría de las modificaciones al dataset (`ds`):

      * **Modificación de Metadatos del Paciente**:
          * `PatientID` se establece con el valor de `DetectorID`.
          * `PatientName` se construye a partir de `StationName` y una traducción de un tag privado de ubicación (`(0x200B, 0x7096)`).
      * **Almacenamiento de la Clasificación**: El resultado mapeado de BAML (ej. "FDT") se guarda en el tag DICOM `ImageComments`, sobrescribiendo cualquier valor previo.
      * **Linealización de Kerma (Modificación de Píxeles)**: Se aplica un método de linealización clave. En lugar de usar una `ModalityLUTSequence`, se calculan y se establecen los tags `RescaleSlope` y `RescaleIntercept` a partir de los datos del CSV de calibración de Kerma. Esto transforma los valores de los píxeles brutos a valores físicos (Kerma en µGy). La `ModalityLUTSequence` original, si existe, se elimina para evitar conflictos.
      * **Almacenamiento de Parámetros de Linealización Física**: Si la función está activada, la pendiente calculada y el tipo de RQA se almacenan en un bloque de tags privados DICOM. Esto no altera los valores de los píxeles.
      * **Saneamiento de Tags**: Se corrigen o establecen tags críticos para asegurar la compatibilidad con el PACS, como `ImageType` y `SpecificCharacterSet` (forzado a `ISO_IR 192` si es necesario).
      * **Generación de Nuevo Nombre de Fichero**: Se crea un nombre de fichero descriptivo basado en varios tags DICOM (ej. `InstanceNumber`, `DetectorID`, `KVP`, etc.).
      * **Guardado del Fichero**: El dataset DICOM modificado se guarda en el directorio de salida con una sintaxis de transferencia no comprimida (`ExplicitVRLittleEndian`).

5.  **Envío a PACS** (`pacs_operations.send_dicom_folder_async`):

      * Una vez que todos los ficheros del directorio de entrada han sido procesados, el sistema inicia el envío de todos los ficheros generados en el directorio de salida.
      * Se establece una conexión (asociación) con el servidor PACS.
      * Cada fichero se envía utilizando el servicio DICOM C-STORE de forma asíncrona.

## 3\. Estructura del Proyecto

El proyecto se organiza en los siguientes ficheros y directorios clave:

```
.
├── main.py                     # Orquestador principal del pipeline
├── config.py                   # Fichero de configuración central
├── utils.py                    # Funciones de utilidad (logging, traducciones, etc.)
├── dicom_processing_pipeline.py# Lógica principal de procesamiento DICOM
├── baml_classification.py      # Interfaz con el servicio de IA BAML
├── linealize.py                # Lógica para la linealización física (opcional)
├── pacs_operations.py          # Lógica para el envío a PACS (C-STORE)
├── requirements.txt            # Dependencias de Python
├── baml_client/                # Cliente BAML generado
├── baml_src/                   # Definiciones BAML (.baml)
├── data/
│   └── linearizacion.csv       # Datos de calibración (VMP vs K_uGy)
├── .env                        # Credenciales de API (no versionado)
├── input_dicom_files/          # Directorio de entrada para ficheros DICOM
└── output_processed_dicom/     # Directorio de salida para ficheros procesados
```

## 4\. Contenido de los Ficheros

A continuación se incluye el contenido completo de los ficheros proporcionados para un contexto detallado.

### `README.md`

````markdown
# Sistema de Procesamiento y Envío DICOM con Clasificación BAML

## 1. Resumen del Proyecto

Este proyecto implementa un pipeline automatizado en Python para el procesamiento avanzado de ficheros DICOM. Las funcionalidades clave incluyen:
    - Lectura y descompresión de imágenes DICOM (incluyendo formatos comprimidos como JPEG Lossless).
    - Modificación de metadatos DICOM (ej. `PatientID`, `PatientName` basados en otros tags y traducciones).
    - Aplicación de una Look-Up Table (LUT) de Kerma a los datos de píxeles para calibración de dosis.
    - Clasificación de imágenes basada en IA utilizando un modelo desplegado con BAML (Boundary AI Markup Language), con mapeo de las clases resultantes a códigos específicos (FDT, MTF, BC, DESCONOCIDA).
    - Almacenamiento de la clasificación obtenida en un tag DICOM configurable (ej. `ImageComments`), sobrescribiendo valores previos.
    - (Opcional) Cálculo de parámetros de linealización física (pendiente, RQA) y almacenamiento de estos en tags privados DICOM sin alterar los píxeles para el PACS.
    - Envío de los ficheros DICOM procesados a un servidor PACS mediante el servicio C-STORE.

El sistema está diseñado de forma modular, utilizando `asyncio` para la gestión de concurrencia en operaciones de E/S (BAML, PACS), y se configura a través de un fichero `config.py` y variables de entorno para el cliente BAML.

## 2. Estructura del Proyecto

El proyecto está organizado en los siguientes módulos y directorios principales:

* `main.py`: Orquestador principal del flujo de trabajo.
* `config.py`: Configuración centralizada (rutas, PACS, parámetros de LUT, etc.).
* `utils.py`: Funciones de utilidad generales (logging, manipulación de ficheros, traducciones).
* `dicom_processing_pipeline.py`: Lógica para leer, descomprimir, modificar metadatos, aplicar LUT Kerma, y sanear tags DICOM.
* `baml_classification.py`: Interfaz con el cliente BAML para la clasificación de imágenes, incluyendo la conversión a PNG base64 y el mapeo de resultados. Implementa reintentos con `tenacity`.
* `linealize.py`: Funciones para calcular parámetros de linealización física y añadirlos a la cabecera DICOM.
* `pacs_operations.py`: Funciones para enviar ficheros DICOM al PACS.
* `requirements.txt`: Lista de dependencias Python.
* `baml_client/`: Cliente BAML generado.
* `baml_src/`: Definiciones BAML (`.baml` files).
* `data/`:
    * `linearizacion.csv`: Fichero CSV con datos de calibración (VMP vs K\_uGy) para la LUT Kerma y/o linealización física.
* `.env`: (No versionado) Fichero para almacenar variables de entorno sensibles (ej. claves API para BAML/Gemini).
* `input_dicom_files/`: Directorio de entrada para los ficheros DICOM a procesar.
* `output_processed_dicom/`: Directorio de salida para los DICOMs procesados.
* `logs/`: (Opcional) Directorio donde se guarda el fichero de log `dicom_workflow.log`.

## 3. Requisitos Previos e Instalación

### 3.1. Software
* Python 3.10 o superior.
* Un servidor PACS accesible y configurado para aceptar conexiones C-STORE.
* Acceso a un servicio BAML (Gemini) configurado y con las credenciales necesarias.
* (Opcional para algunas descompresiones DICOM si `pylibjpeg` no es suficiente) Bibliotecas GDCM instaladas a nivel de sistema.

### 3.2. Dependencias Python
Las dependencias se gestionan a través del fichero `requirements.txt`. Para instalar:
1.  Crea y activa un entorno virtual (recomendado):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Linux/macOS
    # .venv\Scripts\activate    # En Windows
    ```
2.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
    El fichero `requirements.txt` debe incluir como mínimo:
    ```
    pydicom>=2.3.0
    pynetdicom>=2.0.2
    numpy>=1.20.0
    pandas>=1.3.0
    Pillow>=9.0.0
    python-dotenv>=0.20.0
    tenacity>=8.2.0
    # Cualquier dependencia específica de tu cliente BAML
    # (ej. baml-py, httpx, pydantic, openai, etc.)
    # Dependencias para descompresión DICOM:
    pylibjpeg
    pylibjpeg-libjpeg
    pylibjpeg-openjpeg
    pylibjpeg-rle
    ```

### 3.3. Configuración del Cliente BAML
1.  Asegúrate de que los directorios `baml_client/` y `baml_src/` estén presentes y contengan el código generado por BAML. Si se hace algún cambio en los ficheros de `baml_src/` para actualizar `baml_client/` ejecutamos:
    ```bash
    baml-cli generate
    ``` 
2.  Crea un fichero `.env` en la raíz del proyecto (`pacs/`) con las variables de entorno necesarias para tu cliente BAML (ej. `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc., según lo requiera tu configuración de BAML con Gemini).
    ```env
    # Ejemplo de .env
    GEMINI_API_KEY="tu_clave_api_openai_o_gemini" 
    ```

## 4. Configuración del Pipeline

Todos los parámetros principales del pipeline se configuran en `config.py`:

1.  **Rutas**:
    * `INPUT_DICOM_DIR`: Especifica la carpeta donde se encuentran los DICOMs a procesar.
    * `OUTPUT_PROCESSED_DICOM_DIR`: Carpeta donde se guardarán los DICOMs procesados.
    * `LOG_FILENAME`: Nombre del fichero de log (se guardará en `BASE_PROJECT_DIR`).
    * `PATH_LUT_CALIBRATION_CSV`: Ruta al CSV con datos `VMP` y `K_uGy` para la LUT Kerma.
    * `PATH_CSV_LINEALIZACION_FISICA`: (Si `ENABLE_PHYSICAL_LINEALIZATION_PARAMS` es `True`) Ruta al CSV para la linealización física.

2.  **Parámetros de LUT Kerma**:
    * `KERMA_SCALING_FACTOR`: Factor para escalar los valores de Kerma.

3.  **Configuración del PACS**:
    * `PACS_IP`, `PACS_PORT`, `PACS_AET` (del servidor PACS).
    * `CLIENT_AET` (de esta aplicación).

4.  **Parámetros de Linealización Física** (si se habilita):
    * `ENABLE_PHYSICAL_LINEALIZATION_PARAMS`: Poner a `True` para activar.
    * `DEFAULT_RQA_TYPE_LINEALIZATION`: RQA por defecto.
    * `RQA_FACTORS_PHYSICAL_LINEALIZATION`: Diccionario de factores RQA.
    * `PRIVATE_CREATOR_ID_LINEALIZATION`: Identificador para el bloque privado DICOM.

5.  **Clasificación BAML**:
    * `DICOM_TAG_FOR_CLASSIFICATION`: Tag donde se guardará la clasificación (ej. "ImageComments" o "SeriesDescription").
    * `CLASSIFICATION_TAG_PREFIX`: (Opcional, si se decide reintroducir) Prefijo para la clasificación. Actualmente, el código escribe solo el valor mapeado.

6.  **Logging**:
    * `LOG_LEVEL`: Nivel de detalle del log (ej. `logging.INFO`, `logging.DEBUG`).

## 5. Ejecución del Pipeline

1.  **Prepara los Ficheros de Entrada**:
    * Coloca los ficheros DICOM que quieres procesar en la carpeta definida por `config.INPUT_DICOM_DIR`.
    * Asegúrate de que el fichero CSV especificado en `config.PATH_LUT_CALIBRATION_CSV` (y `config.PATH_CSV_LINEALIZACION_FISICA` si aplica) exista y contenga los datos en el formato correcto (columnas `VMP` y `K_uGy`).

2.  **Activa el Entorno Virtual**:
    ```bash
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate    # Windows
    ```

3.  **Ejecuta el Script Principal**:
    Desde el directorio raíz del proyecto (`pacs/`):
    ```bash
    python main.py
    ```

4.  **Monitorización y Resultados**:
    * Observa la salida en la consola para ver el progreso y los mensajes de log de alto nivel.
    * Revisa el fichero de log (definido por `config.LOG_FILENAME`, ej. `dicom_workflow.log`) para un seguimiento detallado, incluyendo mensajes DEBUG si así se configuró.
    * Los ficheros DICOM procesados se guardarán en la carpeta `config.OUTPUT_PROCESSED_DICOM_DIR`.
    * Estos ficheros procesados serán luego enviados al servidor PACS configurado. Verifica en tu PACS que las imágenes hayan llegado y que los metadatos sean los correctos.

## 6. Flujo de Trabajo Detallado

El script `main.py` orquesta los siguientes pasos para cada fichero DICOM:

1.  **Lectura y Descompresión** (`dicom_processing_pipeline.read_and_decompress_dicom`):
    * Se lee el fichero DICOM.
    * Si está comprimido, se intenta descomprimir usando `pylibjpeg` o `gdcm`.
    * Se obtiene el dataset Pydicom (`ds`) y el `pixel_array` (con LUTs/Rescale del DICOM original ya aplicados).

2.  **Clasificación BAML** (`baml_classification.obtener_clasificacion_baml`):
    * El `pixel_array` se normaliza y convierte a una imagen PNG codificada en base64.
    * Esta imagen base64 se envía al cliente BAML (`b.ClassifyImage`).
    * La respuesta de BAML (ej. "Type1") se mapea a un código interno (ej. "FDT") y se manejan posibles errores o límites de tasa con `tenacity`.

3.  **Cálculo de Parámetros de Linealización Física** (`linealize.py`, si está activado en `config.py`):
    * Se cargan datos de calibración VMP vs K\_uGy desde un CSV.
    * Se calcula la pendiente de linealización (VMP vs quanta/area) para el RQA especificado.

4.  **Procesamiento Final y Guardado del DICOM** (`dicom_processing_pipeline.process_and_prepare_dicom_for_pacs`):
    * **Modificación de Metadatos**: Se actualizan `PatientID` (desde `DetectorID`) y `PatientName` (usando `StationName` y la ubicación traducida desde `utils.py`).
    * **Almacenamiento de Clasificación BAML**: El resultado mapeado de BAML (ej. "FDT") se escribe en el tag DICOM configurado (ej. `ImageComments`), sobrescribiendo cualquier valor anterior.
    * **Almacenamiento de Parámetros de Linealización Física**: Si se calcularon, la pendiente y el RQA se escriben en un bloque privado DICOM en la cabecera del `ds` (no se modifican los píxeles con esta linealización).
    * **Aplicación de LUT Kerma**: Se aplica la LUT Kerma (cargada desde el CSV de `config.py`) a los datos de píxeles del `ds`. Esto incluye neutralizar `RescaleSlope/Intercept`, crear la `ModalityLUTSequence` y ajustar `WindowCenter/Width`.
    * **Saneamiento de Tags Críticos**: Se asegura que `ImageType` tenga valores válidos y que `SpecificCharacterSet` esté presente y sea robusto (ej. "ISO\_IR 192") para evitar problemas de codificación en el envío a PACS.
    * **Guardado**: Se genera un nuevo nombre de fichero y el `ds` modificado se guarda en `config.OUTPUT_PROCESSED_DICOM_DIR` con sintaxis `ExplicitVRLittleEndian`.

5.  **Envío a PACS** (`pacs_operations.py`):
    * Una vez procesados todos los ficheros, `main.py` instruye a `pacs_operations.send_dicom_folder_async` para enviar todos los ficheros del directorio de salida al servidor PACS.
    * Cada envío se realiza estableciendo una asociación DICOM, proponiendo contextos de presentación y usando el servicio C-STORE. Las operaciones de red se manejan de forma asíncrona usando `asyncio.to_thread`.

## 7. Verificación de Resultados

Tras la ejecución:
* **Logs**: Revisa `dicom_workflow.log` y la salida de consola para errores o warnings.
* **Directorio de Salida**: Inspecciona los ficheros en `config.OUTPUT_PROCESSED_DICOM_DIR`.
* **Cabeceras DICOM**: Usa un visor DICOM o un script Pydicom para verificar los tags modificados:
    * `PatientID`, `PatientName`
    * El tag de clasificación (ej. `ImageComments`) con el valor mapeado (ej. "FDT").
    * Si la linealización física fue activada: los tags privados con la pendiente y RQA.
    * `ModalityLUTSequence`, `LUTDescriptor`, `LUTExplanation`, `WindowCenter`, `WindowWidth` (reflejando la LUT Kerma).
    * `RescaleSlope=1`, `RescaleIntercept=0`, `RescaleType` ausente.
    * `ImageType` con valores saneados.
    * `SpecificCharacterSet`.
    * `TransferSyntaxUID` (debería ser no comprimida).
* **Servidor PACS**: Confirma que las imágenes procesadas se hayan recibido correctamente.

## 8. Consideraciones Futuras y Mantenimiento
* Implementar un manejo de errores más granular y estrategias de reintento más sofisticadas (más allá de `tenacity` para BAML, quizás para el envío a PACS).
* Mejorar la interfaz de usuario (actualmente es un script de línea de comandos).
* Añadir más pruebas unitarias y de integración.
* Gestionar de forma más segura las credenciales/claves API (ej. con `Azure Key Vault`, `HashiCorp Vault` o variables de entorno en el servidor de despliegue).

---
````

### `main.py`

```python
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
    pixel_values_kerma_lut, kerma_values_kerma_lut = lut_kerma_data_tuple
    private_creator_lin_id = getattr(config, 'PRIVATE_CREATOR_ID_LINEALIZATION', "MY_APP_LINFO_DEFAULT") # Asegurar fallback

    output_dicom_filepath = dicom_processing_pipeline.process_and_prepare_dicom_for_pacs(
        ds=ds, 
        clasificacion_baml_mapeada=clasificacion_a_guardar_final,
        pixel_values_calib=pixel_values_kerma_lut,
        kerma_values_calib=kerma_values_kerma_lut,
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
    pixel_cal_kerma, kerma_cal_kerma = dicom_processing_pipeline.load_calibration_data(
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
```

### `dicom_processing_pipeline.py`

```python
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
```

### `linealize.py`

```python
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
```

### `pacs_operations.py`

```python
# pacs_operations.py
import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional 

logger = logging.getLogger(__name__) 

import pydicom
from pynetdicom import AE, debug_logger, evt, build_context

from pynetdicom.sop_class import (
    CTImageStorage, 
    MRImageStorage,
    ComputedRadiographyImageStorage,
    SecondaryCaptureImageStorage,
    DigitalXRayImageStorageForPresentation,
    Verification 
)
try:
    # Intentamos con el nombre que Python sugirió que podría existir
    from pynetdicom.sop_class import StoragePresentationContexts 
    logger.debug("StoragePresentationContexts importado exitosamente.")
except ImportError:
    StoragePresentationContexts = None 
    logger.warning("No se pudo importar StoragePresentationContexts. Los contextos de almacenamiento se añadirán explícitamente.")

from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, generate_uid

# debug_logger(level=logging.DEBUG) # Descomentar para logs detallados

def _create_ae_with_contexts(client_aet_title: str, dicom_dataset: Optional[pydicom.Dataset] = None) -> AE:
    ae = AE(ae_title=client_aet_title)
    transfer_syntaxes_to_propose = [ExplicitVRLittleEndian, ImplicitVRLittleEndian]

    # Añadir explícitamente las SOP Classes de almacenamiento que se quieren soportar
    common_storage_sops = [
        CTImageStorage, MRImageStorage, ComputedRadiographyImageStorage, 
        SecondaryCaptureImageStorage, DigitalXRayImageStorageForPresentation
    ]
    for sop_class in common_storage_sops:
        if sop_class: 
            ae.add_requested_context(sop_class, transfer_syntaxes_to_propose)
    logger.debug("Contextos de almacenamiento comunes añadidos explícitamente.")

    if dicom_dataset and hasattr(dicom_dataset, 'SOPClassUID'):
        try:
            ae.add_requested_context(dicom_dataset.SOPClassUID, transfer_syntaxes_to_propose)
            logger.debug(f"Contexto específico añadido para SOP Class: {dicom_dataset.SOPClassUID.name}")
        except Exception as e_ctx:
            logger.warning(f"No se pudo añadir contexto específico para SOP Class {getattr(dicom_dataset, 'SOPClassUID', 'Desconocida')}: {e_ctx}")
            
    if Verification: 
        ae.add_requested_context(Verification) 
        logger.debug("Contexto de presentación para Verification (C-ECHO) añadido.")
    else:
        logger.error("No se pudo importar la SOP Class 'Verification'. El C-ECHO podría no funcionar.")
    return ae


def _perform_pacs_send_sync(
    ae_instance: AE, 
    filepath_str: str, # filepath_str es el que se pasa a send_c_store
    pacs_config: Dict[str, Any]
) -> bool:
    assoc = None 
    file_basename = Path(filepath_str).name
    pacs_target_info = f"{pacs_config.get('PACS_AET', 'PACS_DESCONOCIDO')}@{pacs_config.get('PACS_IP', 'IP_DESCONOCIDA')}:{pacs_config.get('PACS_PORT', 'PUERTO_DESCONOCIDO')}"
    
    try:
        logger.info(f"Intentando asociar con PACS ({pacs_target_info}) para enviar: {file_basename}")
        assoc = ae_instance.associate(
            pacs_config["PACS_IP"], 
            pacs_config["PACS_PORT"], 
            ae_title=pacs_config["PACS_AET"]
        )
        if assoc.is_established:
            logger.info(f"Asociación establecida con PACS para {file_basename}. Contextos de presentación aceptados: {len(assoc.accepted_contexts)}")
            
            # Verificación de contexto aceptado para la SOP Class del fichero
            sop_class_to_send = None
            is_context_accepted_for_sop_class = False
            try:
                # Leer el fichero para obtener su SOPClassUID real justo antes de enviar
                ds_to_send = pydicom.dcmread(filepath_str, force=True, stop_before_pixels=True, specific_tags=['SOPClassUID'])
                sop_class_to_send = ds_to_send.SOPClassUID
                is_context_accepted_for_sop_class = any(
                    ctx.abstract_syntax == sop_class_to_send for ctx in assoc.accepted_contexts
                )
                if not is_context_accepted_for_sop_class:
                     logger.warning(f"Ningún contexto fue aceptado para la SOP Class específica ({sop_class_to_send.name}) del fichero {file_basename}.")
            except Exception as e_check:
                logger.warning(f"No se pudo leer SOPClassUID de {file_basename} para verificar contexto aceptado antes del envío: {e_check}.")
            
            if not assoc.accepted_contexts: # Chequeo general
                logger.warning(f"Asociación establecida para {file_basename} pero NO SE ACEPTARON CONTEXTOS DE PRESENTACIÓN. El C-STORE probablemente fallará.")
                # No intentar C-ECHO aquí si el objetivo principal es C-STORE y no hay contextos.
                return False
            
            # Si después de todo, no hay un contexto específico aceptado para la imagen, el C-STORE fallará.
            # Pynetdicom internamente seleccionará un contexto; si no hay uno compatible, fallará.
            if not is_context_accepted_for_sop_class and assoc.accepted_contexts:
                 logger.warning(f"Aunque hay contextos aceptados, ninguno coincide con la SOP Class ({str(sop_class_to_send)}) del fichero {file_basename}. El envío podría fallar.")


            # El C-STORE utiliza el filepath_str para leer el fichero de nuevo.
            status_store = assoc.send_c_store(filepath_str) 
            
            if status_store:
                logger.debug(f"Respuesta C-STORE completa para {file_basename}: {status_store}")
                status_value = getattr(status_store, 'Status', None)
                if status_value == 0x0000:
                    logger.info(f"ÉXITO: {file_basename} enviado correctamente a PACS. Estado: 0x{status_value:04X}")
                    return True
                else:
                    error_comment = getattr(status_store, 'ErrorComment', 'Sin comentario de error específico.')
                    status_description = evt.STATUS_KEYWORDS.get(status_value, f'Estado desconocido (0x{status_value:04X})' if status_value is not None else 'Estado no recibido')
                    logger.error(f"FALLO C-STORE para {file_basename}: {status_description}. Comentario: {error_comment}")
                    return False
            else:
                logger.error(f"No se recibió dataset de estado de C-STORE para {file_basename}.")
                return False
        else:
            logger.error(f"No se pudo establecer asociación con PACS ({pacs_target_info}) para {file_basename}.")
            if assoc and hasattr(assoc, 'acceptor') and assoc.acceptor and hasattr(assoc.acceptor, 'primitive'):
                 logger.error(f"  Detalles de la primitiva A-ASSOCIATE-RJ (si es un rechazo): {assoc.acceptor.primitive}")
            return False
    except ConnectionRefusedError:
        logger.error(f"Error de conexión: El PACS ({pacs_target_info}) rechazó la conexión para {file_basename}.")
        return False
    except ValueError as ve: # Capturar el ValueError de "Failed to encode"
        if "Failed to encode" in str(ve):
            logger.error(f"ERROR DE CODIFICACIÓN DEL DATASET para {file_basename} antes del envío: {ve}", exc_info=True)
        else:
            logger.exception(f"ValueError no esperado durante la operación PACS para {file_basename}: {ve}")
        return False
    except Exception as e: 
        logger.exception(f"Excepción no esperada durante la operación PACS para {file_basename}: {e}")
        return False
    finally:
        if assoc and assoc.is_established:
            assoc.release()
            logger.debug(f"Asociación con PACS liberada para {file_basename}.")


async def send_single_dicom_file_async(filepath_str: str, pacs_config: Dict[str, Any]) -> bool:
    filepath = Path(filepath_str)
    if not filepath.is_file():
        logger.error(f"Fichero DICOM para envío a PACS no existe o no es un fichero: {filepath_str}")
        return False
    dicom_dataset_for_context: Optional[pydicom.Dataset] = None
    try:
        dicom_dataset_for_context = pydicom.dcmread(
            filepath_str, 
            force=True, 
            stop_before_pixels=True, 
            specific_tags=['SOPClassUID']
        )
    except Exception as e_read_meta:
        logger.warning(f"No se pudo leer SOPClassUID de {filepath.name} para contexto específico: {e_read_meta}. ")
    
    ae_instance = _create_ae_with_contexts(
        pacs_config.get("AE_TITLE", "MYPYTHONSCU"), 
        dicom_dataset_for_context
    )
    try:
        return await asyncio.to_thread(
            _perform_pacs_send_sync, 
            ae_instance,
            filepath_str, 
            pacs_config
        )
    except Exception as e: 
        logger.error(f"Error en asyncio.to_thread durante el envío PACS de {filepath.name}: {e}", exc_info=True)
        return False

async def send_dicom_folder_async(folder_path_str: str, pacs_config: Dict[str, Any]) -> bool:
    folder_path = Path(folder_path_str)
    if not folder_path.is_dir():
        logger.error(f"La ruta para envío a PACS no es un directorio válido: {folder_path_str}")
        return False
    dicom_files = [f for f in folder_path.glob("*.dcm") if f.is_file()]
    if not dicom_files:
        logger.info(f"No se encontraron archivos .dcm en {folder_path_str} para enviar al PACS.")
        return True
    pacs_target_info = f"{pacs_config.get('PACS_AET', 'PACS_DESCONOCIDO')}@{pacs_config.get('PACS_IP', 'IP_DESCONOCIDA')}:{pacs_config.get('PACS_PORT', 'PUERTO_DESCONOCIDO')}"
    logger.info(f"Iniciando envío de {len(dicom_files)} archivos desde {folder_path_str} al PACS ({pacs_target_info}).")
    tasks = [send_single_dicom_file_async(str(dicom_file), pacs_config) for dicom_file in dicom_files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success_count = 0
    failure_count = 0
    total_files = len(dicom_files)
    for i, res in enumerate(results):
        file_basename = dicom_files[i].name
        if isinstance(res, Exception):
            logger.error(f"Excepción no controlada enviando {file_basename} al PACS (asyncio.gather): {res}", exc_info=res)
            failure_count += 1
        elif res: 
            success_count += 1
        else: 
            failure_count +=1
    logger.info(f"Resultado del envío masivo a PACS desde {folder_path_str}: "
                f"{success_count} de {total_files} exitosos, {failure_count} de {total_files} fallidos.")
    return failure_count == 0

async def _test_pacs_operations():
    class MockConfigTesting:
        PACS_IP = "jupyter.arnau.scs.es"
        PACS_PORT = 11112
        PACS_AET = "DCM4CHEE"
        CLIENT_AET = "MYPYTHONSCU"
        OUTPUT_TEST_DIR_FOR_PACS = Path("output_pacs_test_send_v_anterior") # Nuevo dir
    MockConfigTesting.OUTPUT_TEST_DIR_for_PACS.mkdir(parents=True, exist_ok=True)
    test_dicom_path: Optional[Path] = None
    try:
        file_meta_test = pydicom.Dataset()
        file_meta_test.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage 
        file_meta_test.MediaStorageSOPInstanceUID = generate_uid()
        file_meta_test.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta_test.TransferSyntaxUID = ExplicitVRLittleEndian
        
        ds_test = pydicom.dataset.FileDataset(None, {}, file_meta=file_meta_test, preamble=b"\0" * 128)
        ds_test.is_little_endian = True
        ds_test.is_implicit_VR = False

        ds_test.PatientName = "Test^PACS^Anterior"
        ds_test.PatientID = "TestPACSAnterior01"
        ds_test.StudyInstanceUID = generate_uid()
        ds_test.SeriesInstanceUID = generate_uid()
        ds_test.SOPInstanceUID = file_meta_test.MediaStorageSOPInstanceUID
        ds_test.SOPClassUID = file_meta_test.MediaStorageSOPClassUID
        ds_test.Modality = "CT"
        ds_test.InstanceNumber = "1"
        ds_test.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"] # Asegurar que ImageType sea una lista de strings
        ds_test.SpecificCharacterSet = "ISO_IR 100" # Poner un charset
        ds_test.Rows = 2
        ds_test.Columns = 2
        ds_test.BitsAllocated = 16
        ds_test.BitsStored = 16
        ds_test.HighBit = 15
        ds_test.PixelRepresentation = 0
        ds_test.SamplesPerPixel = 1
        ds_test.PhotometricInterpretation = "MONOCHROME2"
        import numpy as np 
        ds_test.PixelData = np.array([[100, 200], [300, 400]], dtype=np.uint16).tobytes()

        test_dicom_path_str = str(MockConfigTesting.OUTPUT_TEST_DIR_FOR_PACS / f"test_ct_anterior_{ds_test.SOPInstanceUID[:12]}.dcm")
        test_dicom_path = Path(test_dicom_path_str)
        
        pydicom.dcmwrite(test_dicom_path_str, ds_test, write_like_original=False) 
        logger.info(f"Archivo DICOM de prueba (anterior) creado en: {test_dicom_path_str}")
        
        pacs_config_for_test = {
            "PACS_IP": MockConfigTesting.PACS_IP,
            "PACS_PORT": MockConfigTesting.PACS_PORT,
            "PACS_AET": MockConfigTesting.PACS_AET,
            "AE_TITLE": MockConfigTesting.CLIENT_AET 
        }
        logger.info(f"Iniciando prueba de envío (anterior) a PACS: {pacs_config_for_test['PACS_AET']}...")
        success_single = await send_single_dicom_file_async(test_dicom_path_str, pacs_config_for_test)
        print(f"Resultado envío único (anterior): {'Éxito' if success_single else 'Fallo'}")

    except Exception as e:
        logger.error(f"Error en la prueba de pacs_operations (anterior): {e}", exc_info=True)
    finally:
        import shutil
        if test_dicom_path and test_dicom_path.exists():
            try: test_dicom_path.unlink(missing_ok=True)
            except Exception as e_del: logger.warning(f"No se pudo eliminar {test_dicom_path}: {e_del}")
        if MockConfigTesting.OUTPUT_TEST_DIR_FOR_PACS.exists():
            try: 
                shutil.rmtree(MockConfigTesting.OUTPUT_TEST_DIR_FOR_PACS)
            except Exception as e_del_dir: logger.warning(f"No se pudo eliminar el dir {MockConfigTesting.OUTPUT_TEST_DIR_FOR_PACS}: {e_del_dir}")
        logger.info("Limpieza de prueba de pacs_operations (anterior) completada.")

if __name__ == '__main__':
    import asyncio 
    import shutil 
    from pathlib import Path 
    import numpy as np 

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, 
                            format='%(asctime)s - %(name)s [%(threadName)s] - %(levelname)s - %(message)s')
    
    logger.info("Iniciando prueba de pacs_operations.py (versión anterior)...")
    asyncio.run(_test_pacs_operations())
    logger.info("Prueba de pacs_operations.py (versión anterior) finalizada.")
```

### `utils.py`

```python
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
```

### `get_pydicom_path.py`

```python
import pydicom
import os
print(os.path.dirname(pydicom.__file__))
```