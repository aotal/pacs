import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from PIL import Image as PilImage
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
import pydicom
from pydicom.filewriter import dcmwrite
from pydicom.uid import ExplicitVRLittleEndian
import uvicorn

import numpy as np

try:
    import api_config as config
    import dicom_processing_pipeline
    import linealize
    import pacs_operations
    import pandas as pd
    from utils import configurar_logging_aplicacion, clean_filename_part
except ImportError as e:
    print(f"Error CRITICAL importando modulos: {e}")
    exit()

import joblib
import torch
import torchvision.models as models
import torchvision.transforms as transforms

app = FastAPI(
    title="DICOM Processing API",
    description="An API to process DICOM files through a multi-stage pipeline.",
    version="1.0.0",
)

log_file_path = getattr(config, 'BASE_PROJECT_DIR', Path(".")) / "api_workflow.log"
configurar_logging_aplicacion(str(log_file_path), config.LOG_LEVEL, config.LOG_FORMAT)
logger = logging.getLogger(__name__)

MODELS_DIR = config.BASE_PROJECT_DIR / "modelo_final"
CLASSIFIER_MODEL = joblib.load(MODELS_DIR / "clasificador_svc.joblib")
SCALER_MODEL = joblib.load(MODELS_DIR / "escalador.joblib")

FEATURE_EXTRACTOR = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
FEATURE_EXTRACTOR = torch.nn.Sequential(*list(FEATURE_EXTRACTOR.children())[:-1])
FEATURE_EXTRACTOR.eval()

PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LUT_KERMA_DATA = dicom_processing_pipeline.load_calibration_data(str(config.PATH_LUT_CALIBRATION_CSV))
DF_CALIB_LINEALIZACION = None
if getattr(config, 'ENABLE_PHYSICAL_LINEALIZATION_PARAMS', False):
    DF_CALIB_LINEALIZACION = linealize.obtener_datos_calibracion_vmp_k_linealizacion(
        str(getattr(config, 'PATH_CSV_LINEALIZACION_FISICA', config.PATH_LUT_CALIBRATION_CSV))
    )

PACS_CONFIG = {
    "PACS_IP": config.PACS_IP, "PACS_PORT": config.PACS_PORT,
    "PACS_AET": config.PACS_AET, "AE_TITLE": config.CLIENT_AET
}

TEMP_BASE_DIR = Path("temp_api_processing")
TEMP_BASE_DIR.mkdir(exist_ok=True)

def get_temp_dir() -> Path:
    temp_dir = TEMP_BASE_DIR / str(uuid.uuid4())
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

async def save_upload_file(temp_dir: Path, file: UploadFile) -> Path:
    file_path = temp_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path

async def run_phase_1_decompress(input_path: Path, output_dir: Path) -> Path:
    logger.info(f"Phase 1: Decompressing {input_path.name}")
    ds = pydicom.dcmread(str(input_path), force=True)
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()
    detector_id_fn = ds.get('DetectorID', 'NoDetID')
    kvp_fn = ds.get('KVP', 'NoKVP')
    exposure_uas_fn = float(ds.get('ExposureInuAs', 0.0))
    exposure_index_fn = ds.get('ExposureIndex', 'NoIE')
    new_filename_base = (
        f"Img_decompressed_{clean_filename_part(detector_id_fn)}"
        f"_KVP{clean_filename_part(kvp_fn)}"
        f"_mAs{round(exposure_uas_fn / 1000.0, 2)}"
        f"_IE{clean_filename_part(exposure_index_fn)}"
        f"_{ds.SOPInstanceUID}"
    )
    new_filename = new_filename_base[:200] + ".dcm"
    output_filepath = output_dir / new_filename
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dcmwrite(str(output_filepath), ds)
    logger.info(f"Phase 1: Saved to {output_filepath}")
    return output_filepath

async def run_phase_2_classify(input_path: Path) -> str:
    logger.info(f"Phase 2: Classifying {input_path.name}")
    ds = pydicom.dcmread(str(input_path), force=True)
    pixel_array = ds.pixel_array
    min_val, max_val = np.min(pixel_array), np.max(pixel_array)
    img_array_8bit = np.interp(pixel_array, (min_val, max_val), (0, 255)).astype(np.uint8)
    pil_img = PilImage.fromarray(img_array_8bit)
    img_tensor = PREPROCESS_TRANSFORM(pil_img)
    batch_t = torch.unsqueeze(img_tensor, 0)
    with torch.no_grad():
        features = FEATURE_EXTRACTOR(batch_t)
    vector_img = features.squeeze().numpy()
    vector_2d = vector_img.reshape(1, -1)
    vector_escalado = SCALER_MODEL.transform(vector_2d)
    prediccion = CLASSIFIER_MODEL.predict(vector_escalado)
    clase_predicha = prediccion[0]
    logger.info(f"Phase 2: Predicted class '{clase_predicha}'")
    return clase_predicha

async def run_phase_3_process(input_path: Path, classification: str, output_dir: Path) -> Path:
    logger.info(f"Phase 3: Processing {input_path.name} as '{classification}'")
    ds = pydicom.dcmread(str(input_path), force=True)
    slope_linealizacion = None
    rqa_type_para_tags = None
    if DF_CALIB_LINEALIZACION is not None:
        rqa_type = getattr(config, 'DEFAULT_RQA_TYPE_LINEALIZATION', "RQA5")
        rqa_factors = getattr(config, 'RQA_FACTORS_PHYSICAL_LINEALIZATION', {})
        slope_linealizacion = linealize.calculate_linearization_slope(
            calibration_df=DF_CALIB_LINEALIZACION, rqa_type=rqa_type, rqa_factors_dict=rqa_factors
        )
        if slope_linealizacion:
            rqa_type_para_tags = rqa_type
    pixel_cal, kerma_cal = LUT_KERMA_DATA
    output_filepath = dicom_processing_pipeline.process_and_prepare_dicom_for_pacs(
        ds=ds, clasificacion_baml_mapeada=classification,
        pixel_values_calib=pixel_cal, kerma_values_calib=kerma_cal,
        output_base_dir=output_dir, original_filename=input_path.name,
        linearization_slope_param=slope_linealizacion, rqa_type_param=rqa_type_para_tags,
        private_creator_id_linealizacion=getattr(config, 'PRIVATE_CREATOR_ID_LINEALIZATION', "API_LINFO")
    )
    if not output_filepath:
        raise HTTPException(status_code=500, detail="Phase 3: Failed to process and save DICOM file.")
    logger.info(f"Phase 3: Saved to {output_filepath}")
    return output_filepath

async def run_phase_4_send_pacs(input_path: Path) -> bool:
    logger.info(f"Phase 4: Sending {input_path.name} to PACS")
    success = await pacs_operations.send_single_dicom_file_async(str(input_path), PACS_CONFIG)
    logger.info(f"Phase 4: PACS send status: {'Success' if success else 'Failure'}")
    return success

@app.post("/decompress/", response_class=FileResponse)
async def decompress_dicom_endpoint(temp_dir: Path = Depends(get_temp_dir), file: UploadFile = File(...)):
    input_path = await save_upload_file(temp_dir, file)
    try:
        output_path = await run_phase_1_decompress(input_path, temp_dir)
        return FileResponse(path=output_path, media_type='application/dicom', filename=output_path.name)
    except Exception as e:
        logger.error(f"Error in /decompress endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/", response_class=JSONResponse)
async def classify_dicom_endpoint(temp_dir: Path = Depends(get_temp_dir), file: UploadFile = File(...)):
    input_path = await save_upload_file(temp_dir, file)
    try:
        classification = await run_phase_2_classify(input_path)
        return JSONResponse(content={"classification": classification, "filename": file.filename})
    except Exception as e:
        logger.error(f"Error in /classify endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/", response_class=FileResponse)
async def process_dicom_endpoint(
    temp_dir: Path = Depends(get_temp_dir),
    classification: str = Form(...),
    file: UploadFile = File(...)
):
    input_path = await save_upload_file(temp_dir, file)
    try:
        output_path = await run_phase_3_process(input_path, classification, temp_dir)
        return FileResponse(path=output_path, media_type='application/dicom', filename=output_path.name)
    except Exception as e:
        logger.error(f"Error in /process endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send_pacs/", response_class=JSONResponse)
async def send_pacs_endpoint(temp_dir: Path = Depends(get_temp_dir), file: UploadFile = File(...)):
    input_path = await save_upload_file(temp_dir, file)
    try:
        success = await run_phase_4_send_pacs(input_path)
        return JSONResponse(content={"pacs_send_success": success, "filename": file.filename})
    except Exception as e:
        logger.error(f"Error in /send_pacs endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/full_process/", response_class=JSONResponse)
async def full_process_endpoint(temp_dir: Path = Depends(get_temp_dir), file: UploadFile = File(...)):
    input_path = await save_upload_file(temp_dir, file)
    try:
        decompressed_path = await run_phase_1_decompress(input_path, temp_dir)
        classification = await run_phase_2_classify(decompressed_path)
        processed_path = await run_phase_3_process(decompressed_path, classification, temp_dir)
        pacs_success = await run_phase_4_send_pacs(processed_path)
        return JSONResponse(content={
            "message": "Full process completed.",
            "filename": file.filename,
            "classification": classification,
            "pacs_send_success": pacs_success,
        })
    except Exception as e:
        logger.error(f"Error in /full_process endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=False)
