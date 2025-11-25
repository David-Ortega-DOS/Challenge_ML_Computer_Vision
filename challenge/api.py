import fastapi
import os
import json
from pathlib import Path
from ultralytics import YOLO
from fastapi import UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT_DIR / "artifacts" / "model.onnx"
PT_PATH = ROOT_DIR / "artifacts" / "model_best.pt"

IMGSZ = 896
CONF_TH = 0.25 
IOU_TH = 0.45 
DEVICE = 'cpu'

MODEL = None
CLASSES_META = {"names": ["error"]} 

print("[DEBUG] Iniciando carga del modelo...")
print(f"[DEBUG] Ruta ONNX_PATH: {ONNX_PATH}")
print(f"[DEBUG] Ruta PT_PATH: {PT_PATH}")
print(f"[DEBUG] Dispositivo configurado: {DEVICE}")

try:
    # if ONNX_PATH.exists():
    #     print(f"[DEBUG] Intentando cargar modelo ONNX desde {ONNX_PATH}")
    #     try:
    #         MODEL = YOLO(str(ONNX_PATH)) 
    #         print("Modelo cargado: ONNX para inferencia rápida.")
    #     except Exception as e:
    #         print(f"Fallo al cargar ONNX: {e}. Intentando con PyTorch (.pt)...")
    #         MODEL = None
    # else:
    #     print("[DEBUG] Archivo ONNX no encontrado.")

    # if MODEL is None and PT_PATH.exists():
    #     print(f"[DEBUG] Intentando cargar modelo PyTorch desde {PT_PATH}")
    #     try:
    #         MODEL = YOLO(str(PT_PATH), device=DEVICE)
    #         print("Modelo cargado: PyTorch (.pt) forzado a CPU.")
    #     except Exception as e:
    #         print(f"Fallo al cargar PT: {e}")
    #         MODEL = None
    if PT_PATH.exists():
        try:
            MODEL = YOLO(str(PT_PATH)) 
            print("Modelo cargado: PyTorch (.pt) forzado a CPU.")
        except Exception as e:
            print(f"Fallo al cargar PT: {e}")
            MODEL = None
    if MODEL is None and ONNX_PATH.exists():
        try:
            MODEL = YOLO(str(ONNX_PATH)) 
            print("Modelo cargado: ONNX para inferencia rápida (FALLBACK).")
        except Exception as e:
            print(f"Fallo al cargar ONNX: {e}. Descartando modelo ONNX.")
            MODEL = None

    elif MODEL is None:
        print("[DEBUG] Archivo PyTorch (.pt) no encontrado.")

    if MODEL is None:
        raise FileNotFoundError("Ningún artefacto de modelo válido pudo ser cargado.")
    
    CLASSES_META['names'] = MODEL.names
    
    if len(CLASSES_META['names']) != 17:
        print(f"[WARNING] Modelo cargado con {len(CLASSES_META['names'])} clases, se esperaban 17.")
        print(f"[DEBUG] Clases del modelo: {CLASSES_META['names']}")
    else:
        print("[DEBUG] Modelo cargado con el número esperado de clases (17).")


except Exception as e:
    MODEL = None
    CLASSES_META = {"names": ["error"]}
    print(f" Error FATAL al iniciar la API: {e}")
    print("La API se iniciará, pero /predict fallará (Error 503).")


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
        raise HTTPException(status_code=503, detail="Model artifact missing or failed to load.")
        
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
            conf=CONF_TH, 
            iou=IOU_TH, 
            imgsz=IMGSZ, 
            verbose=False
        )
        
    except Exception as e:
        print(f"Error durante image processing or YOLO inference: {e}")
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


    return PredictionResponse(
        status="success",
        num_detections=len(detections),
        detections=detections
    )