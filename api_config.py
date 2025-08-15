# api_config.py
# Configuration specifically for the FastAPI application.

import logging
from pathlib import Path

# --- Base Project Directory ---
# Assumes api_config.py is in the project root.
BASE_PROJECT_DIR = Path(__file__).resolve().parent

# --- Logging Configuration ---
LOG_FILENAME = "api_workflow.log"
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# --- Data Paths ---
# Path to the CSV file for Kerma LUT calibration and/or physical linearization.
PATH_LUT_CALIBRATION_CSV = BASE_PROJECT_DIR / "data" / "linearizacion.csv"
PATH_CSV_LINEALIZACION_FISICA = BASE_PROJECT_DIR / "data" / "linearizacion.csv"

# --- Kerma LUT Configuration ---
KERMA_SCALING_FACTOR = 100.0

# --- PACS Configuration ---
# These should be replaced with the actual values for the production environment.
PACS_IP = "localhost"
PACS_PORT = 11112
PACS_AET = "DCM4CHEE"
CLIENT_AET = "MYAPI_SCU" # Using a different AET for the API

# --- Physical Linearization Parameters ---
ENABLE_PHYSICAL_LINEALIZATION_PARAMS = False
DEFAULT_RQA_TYPE_LINEALIZATION = "RQA5"
RQA_FACTORS_PHYSICAL_LINEALIZATION: dict[str, float] = {
    "RQA3": 0.000085,
    "RQA5": 0.000123,
    "RQA7": 0.000250,
    "RQA9": 0.000456,
}
PRIVATE_CREATOR_ID_LINEALIZATION = "MIAPP_LINFO_V1"

# --- Application Parameters ---
DICOM_TAG_FOR_CLASSIFICATION = "ImageComments"
CLASSIFICATION_TAG_PREFIX = "QC_Class:"
