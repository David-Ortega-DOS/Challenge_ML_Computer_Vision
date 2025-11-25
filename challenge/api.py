import fastapi
import os
import json
from pathlib import Path
from ultralytics import YOLO
from fastapi import UploadFile, File, HTTPException
from fastapi import Form
from pydantic import BaseModel, Field
import numpy as np
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT_DIR / "artifacts" / "model.onnx"
PT_PATH = ROOT_DIR / "artifacts" / "model_best.pt"

IMGSZ = 896
CONF_TH = 0.25 
IOU_TH = 0.5 
DEVICE = 'cpu'

MODEL = None
CLASSES_META = {"names": ["error"]} 

print("[DEBUG] Iniciando carga del modelo...")
print(f"[DEBUG] Ruta ONNX_PATH: {ONNX_PATH}")
print(f"[DEBUG] Ruta PT_PATH: {PT_PATH}")
print(f"[DEBUG] Dispositivo configurado: {DEVICE}")

try:

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
async def post_predict(
    file: UploadFile = File(...),
    label: UploadFile = File(None)  # Etiqueta opcional
) -> PredictionResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot run inference.")

    image_bytes = await file.read()
    label_data = None

    if label is not None:
        label_data = await label.read()
        print(f"[DEBUG] Etiqueta recibida: {label.filename}")

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("Invalid image file or unsupported format.")

        # Parámetros de inferencia
        print("[DEBUG] Parámetros de inferencia:")
        print(f"  CONF_TH: {CONF_TH}")
        print(f"  IOU_TH: {IOU_TH}")
        print(f"  IMGSZ: {IMGSZ}")

        # Verificar el formato de la imagen procesada
        print("[DEBUG] Procesando imagen para inferencia...")
        print(f"  Tamaño de la imagen recibida: {img_bgr.shape if img_bgr is not None else 'None'}")
        print(f"[DEBUG] Imagen procesada: {file.filename}")

        # Depurar las detecciones generadas por el modelo
        print("[DEBUG] Ejecutando inferencia con el modelo...")
        results_list = MODEL.predict(
            source=img_bgr, 
            conf=CONF_TH, 
            iou=IOU_TH, 
            imgsz=IMGSZ, 
            verbose=False
        )

        if results_list:
            results = results_list[0]
            if results.boxes is not None:
                print("[DEBUG] Detecciones generadas por el modelo:")
                for box, conf, cls_id in zip(
                    results.boxes.xyxy.cpu().numpy(),
                    results.boxes.conf.cpu().numpy(),
                    results.boxes.cls.cpu().numpy().astype(int)
                ):
                    print(f"  Clase: {cls_id}, Confianza: {conf}, Caja: {box}")
            else:
                print("[DEBUG] No se generaron cajas de detección.")
        else:
            print("[DEBUG] No se generaron resultados de inferencia.")

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

    # Procesar etiquetas si se proporcionaron
    if label_data:
        print("[DEBUG] Procesando etiquetas proporcionadas...")
        print(f"[DEBUG] Contenido de la etiqueta recibida:\n{label_data.decode()}\n")
        labels = [list(map(float, line.strip().split())) for line in label_data.decode().splitlines()]
        print(f"[DEBUG] Número de etiquetas recibidas: {len(labels)}")

        # Obtener dimensiones de la imagen procesada
        H, W = img_bgr.shape[:2]  # Alto y ancho de la imagen

        # Comparar detecciones con etiquetas
        matched = 0
        total = len(labels)
        matched_labels = set()  # Para evitar que las etiquetas se cuenten varias veces

        for label in labels:
            cls_id, cx, cy, w, h = label

            # Convertir las cajas de las etiquetas a coordenadas absolutas
            gt_box = [
                (cx - w / 2) * W, (cy - h / 2) * H,  # x_min, y_min
                (cx + w / 2) * W, (cy + h / 2) * H   # x_max, y_max
            ]

            print(f"[DEBUG] Etiqueta procesada (absoluta): Clase {cls_id}, Caja: {gt_box}")

            best_iou = 0.0
            for det in detections:
                det_box = det.box
                iou = iou_xyxy(det_box, gt_box)
                print(f"[DEBUG] Comparando con detección: {det_box}, IoU: {iou}")
                if iou >= IOU_TH and tuple(gt_box) not in matched_labels:
                    best_iou = max(best_iou, iou)

            if best_iou >= IOU_TH:
                matched += 1
                matched_labels.add(tuple(gt_box))  # Marcar la etiqueta como emparejada

        recall = matched / total if total > 0 else 0.0
        print(f"[DEBUG] Recall@IoU>={IOU_TH}: {recall:.3f} ({matched}/{total})")

    return PredictionResponse(
        status="success",
        num_detections=len(detections),
        detections=detections
    )


def iou_xyxy(box_a, box_b):
    """
    Calcula el IoU (Intersection over Union) entre dos cajas en formato [x_min, y_min, x_max, y_max].
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = box_a_area + box_b_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0