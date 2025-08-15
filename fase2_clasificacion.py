# fase2_clasificacion.py

import logging
from pathlib import Path
import shutil
import joblib
import pydicom
import numpy as np
from PIL import Image as PilImage

# --- Importaciones para el modelo ---
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_PROJECT_DIR = Path.cwd()
# Directorio de entrada: donde la fase 1 dejó los ficheros
INPUT_DIR = BASE_PROJECT_DIR / "f1_descomprimidos"
# Directorio raíz para los ficheros clasificados
OUTPUT_DIR_CLASIFICADOS = BASE_PROJECT_DIR / "f2_clasificados"
# Directorio donde se encuentra el modelo guardado
MODEL_DIR = BASE_PROJECT_DIR / "modelo_final"

# Clases esperadas (nombres de las carpetas de salida)
CLASES = ["FDT", "MTF", "TOR"]

# --- PASO 1: CARGAR MODELOS Y LÓGICA DE EXTRACCIÓN DE VECTORES ---
# Reutilizamos la lógica del script de inferencia para coherencia.

logger.info("Cargando modelo ResNet50 para extracción de características...")
try:
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
    feature_extractor.eval()
    logger.info("Modelo ResNet50 cargado.")
except Exception as e:
    logger.error(f"No se pudo cargar el modelo ResNet50. Error: {e}")
    exit()

# Definimos la transformación necesaria para ResNet50
preprocess_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extraer_vector_de_un_dicom(filepath: Path, crop_pixels: int = 20) -> np.ndarray:
    """
    Toma la ruta de un fichero DICOM, lo procesa y devuelve su vector de características.
    """
    logger.debug(f"Procesando imagen: {filepath.name}")
    ds = pydicom.dcmread(str(filepath), force=True)
    pixel_array = ds.pixel_array

    min_val, max_val = np.min(pixel_array), np.max(pixel_array)
    img_array_8bit = np.interp(pixel_array, (min_val, max_val), (0, 255)).astype(np.uint8) if max_val > min_val else np.zeros_like(pixel_array, dtype=np.uint8)
    
    pil_img = PilImage.fromarray(img_array_8bit)
    
    bbox = pil_img.getbbox()
    if bbox: pil_img = pil_img.crop(bbox)
    
    w, h = pil_img.size
    if w > crop_pixels * 2 and h > crop_pixels * 2:
        pil_img = pil_img.crop((crop_pixels, crop_pixels, w - crop_pixels, h - crop_pixels))
    
    if pil_img.size[0] == 0 or pil_img.size[1] == 0:
        raise ValueError("La imagen no tiene contenido para procesar después del recorte.")

    # Aplicar transformaciones y extraer vector
    img_tensor = preprocess_transform(pil_img)
    batch_t = torch.unsqueeze(img_tensor, 0)
    with torch.no_grad():
        features = feature_extractor(batch_t)
    
    return features.squeeze().numpy()

# --- PASO 2: CLASIFICAR CADA IMAGEN Y MOVERLA A SU CARPETA ---
def ejecutar_fase2_clasificacion():
    """
    Función principal que clasifica todos los ficheros de un directorio
    y los mueve a sus respectivas carpetas de clase.
    """
    logger.info("===== INICIO FASE 2: CLASIFICACIÓN DE IMÁGENES DESCOMPRIMIDAS =====")

    # Comprobar si el directorio de entrada existe
    if not INPUT_DIR.exists():
        logger.error(f"El directorio de entrada '{INPUT_DIR}' no existe.")
        logger.error("Asegúrate de haber ejecutado la fase 1 primero.")
        return

    # Cargar el clasificador y el escalador guardados
    try:
        logger.info(f"Cargando modelo desde {MODEL_DIR}...")
        model = joblib.load(MODEL_DIR / "clasificador_svc.joblib")
        scaler = joblib.load(MODEL_DIR / "escalador.joblib")
        logger.info("Modelo y escalador cargados correctamente.")
    except FileNotFoundError:
        logger.error(f"Error: No se encontró el modelo o el escalador en '{MODEL_DIR}'.")
        logger.error("Asegúrate de que los ficheros 'clasificador_svc.joblib' y 'escalador.joblib' existen.")
        return

    # Crear directorios de salida
    if OUTPUT_DIR_CLASIFICADOS.exists():
        shutil.rmtree(OUTPUT_DIR_CLASIFICADOS)
        logger.warning(f"Se ha eliminado el directorio de salida existente: {OUTPUT_DIR_CLASIFICADOS}")
    
    for clase in CLASES:
        (OUTPUT_DIR_CLASIFICADOS / clase).mkdir(parents=True, exist_ok=True)
    
    # Listar ficheros DICOM en el directorio de entrada
    dicom_files = list(INPUT_DIR.glob("*.dcm"))
    if not dicom_files:
        logger.warning(f"No se encontraron ficheros .dcm en '{INPUT_DIR}'.")
        return

    logger.info(f"Se encontraron {len(dicom_files)} ficheros para clasificar.")
    
    # Procesar y clasificar cada fichero
    for filepath in dicom_files:
        try:
            # 1. Extraer vector
            vector_img = extraer_vector_de_un_dicom(filepath)
            
            # 2. Escalar
            vector_2d = vector_img.reshape(1, -1)
            vector_escalado = scaler.transform(vector_2d)
            
            # 3. Predecir
            prediccion = model.predict(vector_escalado)
            clase_predicha = prediccion[0]
            
            logger.info(f"Fichero: {filepath.name} -> Clase predicha: '{clase_predicha}'")
            
            # 4. Mover fichero a la carpeta correspondiente
            if clase_predicha in CLASES:
                destino = OUTPUT_DIR_CLASIFICADOS / clase_predicha / filepath.name
                shutil.move(str(filepath), str(destino))
                logger.info(f" -> Movido a: {destino}")
            else:
                logger.warning(f"Clase '{clase_predicha}' no es una de las clases esperadas {CLASES}. El fichero no se moverá.")

        except Exception as e:
            logger.error(f"Error procesando el fichero {filepath.name}: {e}", exc_info=True)

    logger.info("===== FIN FASE 2: CLASIFICACIÓN COMPLETADA =====")


if __name__ == "__main__":
    ejecutar_fase2_clasificacion()