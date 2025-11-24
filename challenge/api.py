import fastapi
import os
import json
from pathlib import Path
from ultralytics import YOLO
from fastapi import UploadFile, File, HTTPException, Body
from pydantic import BaseModel, Field
import numpy as np
import cv2

# --- CONFIGURACIÓN GLOBAL DE ARTEFACTOS ---
ONNX_PATH = Path("artifacts/model.onnx")
PT_PATH = Path("artifacts/model_best.pt")
CLASSES_PATH = Path("artifacts/classes.json")

try:
    MODEL = None

    if ONNX_PATH.exists():
        try:
            MODEL = YOLO(str(ONNX_PATH)) 
            print(" Modelo cargado: ONNX para inferencia rápida.")
        except Exception as e:
            print(f" Fallo fatal al cargar ONNX: {e}. Intentando con PyTorch (.pt)...")
            MODEL = None
            
    if MODEL is None and PT_PATH.exists():
        try:
            MODEL = YOLO(str(PT_PATH))
            print(" Modelo cargado: PyTorch (.pt).")
        except Exception as e:
            print(f" Fallo al cargar PT: {e}")
            MODEL = None

    if MODEL is None:
        raise FileNotFoundError("Ningún artefacto de modelo válido pudo ser cargado.")
    
    with open(CLASSES_PATH, "r") as f:
        CLASSES_META = json.load(f)

    assert len(CLASSES_META['names']) == 17, "Error: La lista de nombres de clase no tiene 17 elementos."

except FileNotFoundError as e:
    MODEL = None
    CLASSES_META = {"names": ["error"]}
    print(f" Error al iniciar la API: {e}")
    print("La API se iniciará, pero /predict fallará (Error 503).")


# --- ESQUEMAS PYDANTIC DE RESPUESTA ---
class Detection(BaseModel):
    box: list[float] = Field(description="[x_min, y_min, x_max, y_max] en píxeles")
    confidence: float = Field(description="Confianza de la detección")
    class_id: int
    class_name: str

class PredictionResponse(BaseModel):
    status: str = "success"
    num_detections: int
    detections: list[Detection]


app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model artifact missing.")
        
    return {
        "status": "model_loaded",
    }


@app.post("/predict", status_code=200, response_model=PredictionResponse)
async def post_predict(file: UploadFile = File(...)) -> PredictionResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot run inference.")

    image_bytes = await file.read()
    
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("Invalid image file or unsupported format.")
        
        results_list = MODEL.predict(
            source=img_bgr, 
            conf=0.25, 
            iou=0.45,       
            imgsz=896,      
            verbose=False
        )
        
    except Exception as e:
        print(f"Error during image processing or YOLO inference: {e}")
        raise HTTPException(status_code=500, detail="Inference failed after image decoding.")

    detections = []
    
    if results_list:
        results = results_list[0]
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            num_classes = len(CLASSES_META['names']) 
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    
                    if 0 <= cls_id < num_classes:
                        detections.append(Detection(
                            box=box.tolist(),
                            confidence=float(conf),
                            class_id=int(cls_id),
                            class_name=CLASSES_META['names'][cls_id]
                        ))
                    else:
                        print(f"DEBUG: Ignorando detección con cls_id fuera de rango: {cls_id}")

    return PredictionResponse(
        status="success",
        num_detections=len(detections),
        detections=detections
    )