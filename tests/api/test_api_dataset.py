import io
import os
import cv2
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from challenge.api import app
from tests.conftest import _img_to_label
from tests.utils_iou import yolo_to_xyxy, iou_xyxy

client = TestClient(app)


def _to_buf(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.mark.slow
def test_api_recall_on_subset(test_images, yolo_labels):
    iou_th = float(os.environ.get("IOU_TH", 0.5))

    matched = 0
    total = 0

    for img_path in test_images:
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        buf = _to_buf(img)

        # Usar _img_to_label para generar la ruta correcta de la etiqueta
        lbl_path = _img_to_label(img_path)

        # Verificar si el archivo de etiquetas existe
        if not os.path.exists(lbl_path):
            print(f"[DEBUG] Archivo de etiquetas no encontrado: {lbl_path}")
            pytest.fail(f"Etiqueta faltante para la imagen: {img_path}")

        # Leer el archivo de etiquetas
        with open(lbl_path, "r") as f:
            label_content = f.read()

        # Enviar imagen y etiqueta al endpoint
        resp = client.post(
            "/predict",
            files={
                "file": ("img.jpg", buf, "image/jpeg"),
                "label": ("label.txt", label_content, "text/plain")
            }
        )

        assert resp.status_code in (200, 400)
        if resp.status_code == 400:
            pytest.skip("The API was started without weights. Exiting with skip.")
        dets = resp.json()["detections"]

        gts = yolo_labels(lbl_path)

        for cid, cx, cy, w, h in gts:
            total += 1
            gt_xyxy = yolo_to_xyxy(cx, cy, w, h, W, H)
            best = 0.0
            for d in dets:
                if d["class_id"] != cid:
                    continue
                bx = (
                    int(d["box"][0]),  # x_min
                    int(d["box"][1]),  # y_min
                    int(d["box"][2]),  # x_max
                    int(d["box"][3])   # y_max
                )
                best = max(best, iou_xyxy(bx, gt_xyxy))
            if best >= iou_th:
                matched += 1

    recall = matched / total if total > 0 else 0.0
    print(f"[API] Recall@IoU>={iou_th}: {recall:.3f} ({matched}/{total})")
    assert recall >= float(os.environ.get("API_MIN_RECALL", 0.10))
