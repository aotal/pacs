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
2.  Crea un fichero `.env` en la raíz del proyecto (`pacs/`) con las variables de entorno necesarias para tu cliente BAML (ej. `OPENAI_API_KEY`, `BAML_PROJECT_ID`, etc., según lo requiera tu configuración de BAML con Gemini).
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