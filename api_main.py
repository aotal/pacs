import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
import zipfile
import io
import base64

from PIL import Image as PilImage
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pydicom
from pydicom.filewriter import dcmwrite
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.pixel_data_handlers.util import apply_voi_lut
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
    print(f"Error CRÍTICO importando módulos: {e}")
    exit()

import joblib
import torch
import torchvision.models as models
import torchvision.transforms as transforms

app = FastAPI(
    title="DICOM Processing API",
    description="Una API para procesar ficheros DICOM a través de un pipeline multi-etapa.",
    version="1.0.0",
)

origins = [
    "http://localhost:9002", # La URL de tu frontend en Firebase Studio
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
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

PACS_CONFIG = {
    "PACS_IP": config.PACS_IP, "PACS_PORT": config.PACS_PORT,
    "PACS_AET": config.PACS_AET, "AE_TITLE": config.CLIENT_AET
}

TEMP_BASE_DIR = Path("temp_api_processing")
TEMP_BASE_DIR.mkdir(exist_ok=True)

# --- Pydantic Models for new endpoint ---
class KermaDataItem(BaseModel):
    mAs: float
    kerma: float

class ClassifiedFile(BaseModel):
    id: int
    name: str
    processedName: str
    class_name: str = Field(..., alias='class')

class LinearizationRequest(BaseModel):
    files: List[ClassifiedFile]
    kerma_data: List[KermaDataItem]
    sdd: float
    sid: float
    session_id: str

def get_temp_dir() -> Path:
    session_id = str(uuid.uuid4())
    temp_dir = TEMP_BASE_DIR / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        pass

def calculate_vmp_in_roi(ds: pydicom.Dataset, roi_size_cm: float = 4.0) -> float:
    # --- CORRECCIÓN: Usar SpatialResolution si PixelSpacing no está disponible ---
    pixel_spacing_val = None
    if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing:
        pixel_spacing_val = ds.PixelSpacing
    elif hasattr(ds, 'SpatialResolution') and ds.SpatialResolution:
        # SpatialResolution es un solo valor (mm), se usa para ambas dimensiones
        pixel_spacing_val = [ds.SpatialResolution, ds.SpatialResolution]

    if not pixel_spacing_val or len(pixel_spacing_val) < 2:
        # Usar ds.filename si está disponible para un mejor mensaje de error
        filename = getattr(ds, 'filename', 'desconocido')
        raise ValueError(f"El fichero {filename} no contiene 'PixelSpacing' ni 'SpatialResolution', etiquetas necesarias para el cálculo del ROI.")

    roi_size_px_x = int((roi_size_cm * 10) / pixel_spacing_val[0])
    roi_size_px_y = int((roi_size_cm * 10) / pixel_spacing_val[1])
    
    center_x = ds.pixel_array.shape[1] // 2
    center_y = ds.pixel_array.shape[0] // 2
    
    half_roi_x = roi_size_px_x // 2
    half_roi_y = roi_size_px_y // 2
    
    roi = ds.pixel_array[
        center_y - half_roi_y : center_y + half_roi_y,
        center_x - half_roi_x : center_x + half_roi_x
    ]
    
    return float(np.mean(roi))

def apply_manual_rescale_tags(ds: pydicom.Dataset, slope: float, intercept: float) -> pydicom.Dataset:
    ds.RescaleSlope = float(slope)
    ds.RescaleIntercept = float(intercept)
    ds.RescaleType = 'K_uGy'
    if 'ModalityLUTSequence' in ds:
        del ds.ModalityLUTSequence
    
    bits_stored = ds.BitsStored
    min_pixel_val = 0
    max_pixel_val = (2**bits_stored) - 1
    min_output_val = min_pixel_val * ds.RescaleSlope + ds.RescaleIntercept
    max_output_val = max_pixel_val * ds.RescaleSlope + ds.RescaleIntercept
    new_ww = max(1.0, max_output_val - min_output_val)
    new_wc = min_output_val + new_ww / 2.0
    ds.WindowCenter = float(new_wc)
    ds.WindowWidth = float(new_ww)
    return ds

def get_8bit_image_array(ds: pydicom.Dataset) -> np.ndarray:
    """Función unificada para obtener una imagen de 8-bit desde un DICOM."""
    try:
        img_8bit = apply_voi_lut(ds.pixel_array, ds)
        if img_8bit.dtype != np.uint8:
            raise ValueError("apply_voi_lut no devolvió uint8")
        return img_8bit
    except Exception:
        pixel_array = ds.pixel_array
        min_val, max_val = np.min(pixel_array), np.max(pixel_array)
        if max_val == min_val:
            return np.zeros(pixel_array.shape, dtype=np.uint8)
        else:
            return np.interp(pixel_array, (min_val, max_val), (0, 255)).astype(np.uint8)

async def run_phase_2_classify(input_path: Path) -> str:
    ds = pydicom.dcmread(str(input_path), force=True)
    # --- CORRECCIÓN: Usar la función unificada para consistencia ---
    img_array_8bit = get_8bit_image_array(ds)
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
    return clase_predicha

def create_thumbnail_b64(ds: pydicom.Dataset, size: tuple[int, int] = (128, 128)) -> str:
    try:
        # --- CORRECCIÓN: Usar la función unificada para consistencia ---
        img_array_8bit = get_8bit_image_array(ds)
        pil_img = PilImage.fromarray(img_array_8bit)
        pil_img.thumbnail(size)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error creando miniatura: {e}", exc_info=True)
        return ""

@app.post("/decompress_and_classify/", response_class=JSONResponse)
async def decompress_and_classify_endpoint(
    files: List[UploadFile] = File(...),
    temp_dir: Path = Depends(get_temp_dir)
):
    async def process_single_file(file: UploadFile) -> Dict[str, Any]:
        original_filename = file.filename
        try:
            input_path = temp_dir / original_filename
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            ds = pydicom.dcmread(str(input_path), force=True)
            if ds.file_meta.TransferSyntaxUID.is_compressed:
                ds.decompress()

            processed_name = f"decom_{uuid.uuid4()}.dcm"
            processed_path = temp_dir / processed_name
            ds.save_as(str(processed_path))

            classification = await run_phase_2_classify(processed_path)
            thumbnail_b64 = create_thumbnail_b64(ds)
            
            kvp = ds.get('KVP', 'N/A')
            exposure_uas = float(ds.get('ExposureInuAs', 0.0))
            mas = round(exposure_uas / 1000.0, 2)
            ie = ds.get('ExposureIndex', 'N/A')

            return {
                "filename": original_filename,
                "processedName": processed_name,
                "classification": classification,
                "image_b64": thumbnail_b64,
                "kvp": kvp, "mas": mas, "ie": ie
            }
        except Exception as e:
            logger.error(f"Error procesando {original_filename}: {e}", exc_info=True)
            return {"filename": original_filename, "classification": "Error"}

    tasks = [process_single_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    return JSONResponse(content={"results": results, "session_id": temp_dir.name})

@app.post("/process_and_linearize/", response_class=JSONResponse)
async def process_and_linearize_endpoint(request: LinearizationRequest):
    session_id = request.session_id
    session_dir = TEMP_BASE_DIR / session_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found.")
        
    output_dir = session_dir / "linearized"
    output_dir.mkdir(exist_ok=True)

    try:
        fdt_files = [f for f in request.files if f.class_name == 'FDT']
        if len(fdt_files) < 2:
            raise HTTPException(status_code=400, detail=f"Se necesitan al menos 2 imágenes FDT. Se encontraron {len(fdt_files)}.")

        kerma_map = {item.mAs: item.kerma for item in request.kerma_data}
        vmp_points, kerma_points = [], []

        for fdt_file in fdt_files:
            file_path = session_dir / fdt_file.processedName
            ds = pydicom.dcmread(str(file_path), force=True)
            vmp = calculate_vmp_in_roi(ds)
            file_mas = round(float(ds.get('ExposureInuAs', 0.0)) / 1000.0, 2)
            
            if file_mas in kerma_map:
                kerma_detector = kerma_map[file_mas]
                kerma_receptor = kerma_detector * (request.sdd**2 / request.sid**2)
                vmp_points.append(vmp)
                kerma_points.append(kerma_receptor)

        if len(vmp_points) < 2:
            raise HTTPException(status_code=400, detail="No se pudieron emparejar suficientes puntos para la regresión.")

        slope, intercept = np.polyfit(vmp_points, kerma_points, 1)

        for file_info in request.files:
            input_path = session_dir / file_info.processedName
            output_path = output_dir / f"linearized_{Path(file_info.processedName).name}"
            ds = pydicom.dcmread(str(input_path))
            ds = apply_manual_rescale_tags(ds, slope, intercept)
            ds.save_as(str(output_path))

        return JSONResponse(content={"message": "Linealización completada.", "session_id": session_id})
    except ValueError as ve:
        logger.error(f"Error de valor durante la linealización: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during linearization for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en la linealización: {str(e)}")

@app.get("/download_processed_files/{session_id}", response_class=StreamingResponse)
async def download_processed_files(session_id: str):
    linearized_dir = TEMP_BASE_DIR / session_id / "linearized"
    if not linearized_dir.is_dir():
        raise HTTPException(status_code=404, detail="No se encontraron ficheros.")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for path in linearized_dir.glob("*.dcm"):
            zip_file.write(path, arcname=path.name)
    zip_buffer.seek(0)
    
    return StreamingResponse(zip_buffer, media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment; filename=ficheros_linealizados.zip"})

@app.post("/send_processed_to_pacs/{session_id}", response_class=JSONResponse)
async def send_processed_to_pacs(session_id: str):
    linearized_dir = TEMP_BASE_DIR / session_id / "linearized"
    if not linearized_dir.is_dir():
        raise HTTPException(status_code=404, detail="No se encontraron ficheros.")
    
    files_to_send = [str(p) for p in linearized_dir.glob("*.dcm")]
    tasks = [pacs_operations.send_single_dicom_file_async(f, PACS_CONFIG) for f in files_to_send]
    results = await asyncio.gather(*tasks)
    
    success_count = sum(1 for r in results if r)
    return JSONResponse(content={"message": f"{success_count}/{len(files_to_send)} ficheros enviados."})

if __name__ == "__main__":
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)
