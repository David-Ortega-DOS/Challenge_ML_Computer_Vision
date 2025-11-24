import fastapi
import os
import json
from pathlib import Path
from ultralytics import YOLO # Necesario para cargar modelos .pt y .onnx
from fastapi import UploadFile, File, HTTPException, Body
from pydantic import BaseModel, Field
import numpy as np
import cv2 # Necesario para procesar la imagen de bytes

# --- CONFIGURACIÓN GLOBAL DE ARTEFACTOS ---
ONNX_PATH = Path("artifacts/model.onnx")
PT_PATH = Path("artifacts/model_best.pt")
CLASSES_PATH = Path("artifacts/classes.json")

# Cargamos el modelo y la metadata UNA SOLA VEZ
try:
    if ONNX_PATH.exists():
        # --- CAMBIO CLAVE: Cargar solo la ruta, sin argumentos de dispositivo ---
        # ONNX Runtime (que se carga internamente) usará la CPU por defecto
        # si no puede usar CUDA (lo que ocurre en tu MX350).
        MODEL = YOLO(str(ONNX_PATH))
        print("Modelo cargado: ONNX (Usando CPU por defecto).")
        
    elif PT_PATH.exists():
        # Fallback al modelo PT, donde sí podemos forzar la CPU
        MODEL = YOLO(str(PT_PATH), device="cpu") 
        print("Modelo cargado: PyTorch (.pt, forzado a CPU).")
    else:
        # Fallback si se ejecuta make api-test sin modelo
        MODEL = None 
        raise FileNotFoundError("Modelo no encontrado. Asegúrese de correr la Parte I.")
    
    with open(CLASSES_PATH, "r") as f:
        CLASSES_META = json.load(f)

except FileNotFoundError as e:
    MODEL = None
    CLASSES_META = {"names": ["error"]}
    print(f"Error al iniciar la API: {e}")
    print("La API se iniciará, pero /predict fallará (Error 503).")


# --- ESQUEMAS PYDANTIC DE RESPUESTA ---

# Estructura de cada detección
class Detection(BaseModel):
    box: list[float] = Field(description="[x_min, y_min, x_max, y_max] en píxeles")
    confidence: float = Field(description="Confianza de la detección")
    class_id: int
    class_name: str

# Estructura de la respuesta /predict
class PredictionResponse(BaseModel):
    status: str = "success"
    num_detections: int
    detections: list[Detection]


app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    # Retorna "model_loaded" solo si el modelo se cargó exitosamente
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model artifact missing.")
        
    return {
        "status": "model_loaded",
    }


@app.post("/predict", status_code=200, response_model=PredictionResponse)
async def post_predict(file: UploadFile = File(...)) -> PredictionResponse:
    # 1. Verificar el estado del modelo
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot run inference.")

    # 2. Leer el archivo de imagen (como bytes)
    image_bytes = await file.read()
    
    # 3. Ejecutar Inferencia
    try:
        # 1. Convertir los bytes a un array numpy (buffer)
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # 2. Decodificar el array numpy en una imagen OpenCV (BGR)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("Invalid image file or unsupported format.")
        
        # YOLO.predict() acepta el array numpy decodificado (BGR)
        results_list = MODEL.predict(
            source=img_bgr, # Pasar el array NumPy, no los bytes
            conf=0.25, 
            iou=0.45,       
            imgsz=800,      
            verbose=False
        )
        
    except Exception as e:
        # El 500 es causado por un error en la conversión o decodificación
        print(f"Error during image processing or YOLO inference: {e}")
        # Usar HTTPException 400 si la imagen es inválida, 500 si es un error del modelo.
        raise HTTPException(status_code=500, detail="Inference failed after image decoding.")

    # 4. Procesar y Formatear la Respuesta
    detections = []
    
    if results_list:
        results = results_list[0]
        
        if results.boxes is not None:
            # Extraer los tensores de la GPU (si aplica) o CPU
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detections.append(Detection(
                    # Convertir np.array a lista para JSON
                    box=box.tolist(), 
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=CLASSES_META['names'][cls_id]
                ))

    return PredictionResponse(
        status="success",
        num_detections=len(detections),
        detections=detections
    )