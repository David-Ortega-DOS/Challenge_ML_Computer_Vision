import json
import os

import cv2
from ultralytics import YOLO

# --- VARIABLES DE CONFIGURACIÓN ---
IMG_PATH = "/mnt/d/Github/Challenge_ML_Computer_Vision/data/test/images/H-220927_E11_Y-49_011_-_MP4-19_jpg.rf.f25628e47f83b8f020dd242198fe7ba9.jpg"
ONNX_MODEL_PATH = "artifacts/model.onnx"

try:
    model = YOLO(ONNX_MODEL_PATH)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# --- INFERENCIA ---
print(f"Iniciando inferencia en: {IMG_PATH}")
results = model.predict(
    source=IMG_PATH, conf=0.01, iou=0.45, imgsz=896, device="cpu", verbose=True
)

# --- ANÁLISIS DE RESULTADOS ---
if results and results[0].boxes is not None:
    boxes = results[0].boxes
    detections = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)

    print(f"\n Detecciones encontradas: {len(detections)}")
    print("-----------------------------------")

    for i in range(len(detections)):
        if confidences[i] > 0.25:
            print(
                f"Clase ID: {class_ids[i]}, Confianza: {confidences[i]:.2f}, Box: {detections[i].astype(int)}"
            )

else:
    print(" No se encontraron detecciones con conf > 0.01.")
