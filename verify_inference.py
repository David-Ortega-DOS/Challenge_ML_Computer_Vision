import cv2
import json
from ultralytics import YOLO
import os

# --- VARIABLES DE CONFIGURACIÓN ---
# RUTA a la imagen que obtuviste del debugging de pytest
IMG_PATH = '/mnt/d/Github/Challenge_ML_Computer_Vision/data/test/images/H-220927_E11_Y-49_011_-_MP4-19_jpg.rf.f25628e47f83b8f020dd242198fe7ba9.jpg' 
ONNX_MODEL_PATH = 'artifacts/model.onnx'

# Carga del modelo (forzado a CPU para evitar el error de la MX350)
try:
    # Usamos la misma lógica de carga que en api.py
    model = YOLO(ONNX_MODEL_PATH)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# --- INFERENCIA ---
print(f"Iniciando inferencia en: {IMG_PATH}")
results = model.predict(
    source=IMG_PATH, 
    conf=0.01, # Usamos un umbral MUY bajo para ver si detecta ALGO
    iou=0.45,
    imgsz=800,
    device='cpu',
    verbose=True
)

# --- ANÁLISIS DE RESULTADOS ---
if results and results[0].boxes is not None:
    boxes = results[0].boxes
    detections = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    
    print(f"\n✅ Detecciones encontradas: {len(detections)}")
    print("-----------------------------------")
    
    # Imprimir las detecciones con alta confianza
    for i in range(len(detections)):
        if confidences[i] > 0.25: # Solo si la confianza es media-alta
            print(f"Clase ID: {class_ids[i]}, Confianza: {confidences[i]:.2f}, Box: {detections[i].astype(int)}")

else:
    print("❌ No se encontraron detecciones con conf > 0.01.")