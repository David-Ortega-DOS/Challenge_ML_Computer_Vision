import os
import glob
import yaml
import random
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_YAML_PATH = PROJECT_ROOT / "data" / "data.yaml"


def _load_data_cfg():
    assert (
        DATA_YAML_PATH.exists()
    ), f"data.yaml no encontrado en {DATA_YAML_PATH.resolve()}"
    with open(DATA_YAML_PATH, "r") as f:
        return yaml.safe_load(f)


def _collect_split_images(img_dir: str):
    full_img_dir = PROJECT_ROOT / img_dir
    if not full_img_dir.is_dir():
        print(f"DEBUG: Directorio no encontrado: {full_img_dir}")
        return []
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob.glob(str(full_img_dir / ext)))
    return sorted(paths)


def _img_to_label(img_path: str) -> str:
    lbl = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    lbl = os.path.splitext(lbl)[0] + ".txt"
    return lbl


@pytest.fixture(scope="session")
def data_cfg():
    return _load_data_cfg()


@pytest.fixture(scope="session")
def class_names(data_cfg):
    return data_cfg.get("names", [])


@pytest.fixture(scope="session")
def test_images(data_cfg):
    test_dir = data_cfg.get("test") or data_cfg.get("val")
    assert test_dir
    imgs = _collect_split_images(test_dir)
    assert imgs
    k = int(os.environ.get("TEST_SAMPLE_SIZE", 6))
    random.seed(7)

    ##DEBUG
    selected_images = random.sample(imgs, min(k, len(imgs)))
    print(f"\n[DEBUG] Imagen de Prueba Clave: {selected_images[0]}")

    # Generar la etiqueta correspondiente a la imagen de prueba clave
    test_label = _img_to_label(selected_images[0])
    print(f"[DEBUG] Etiqueta de Prueba Clave: {test_label}")

    return random.sample(imgs, min(k, len(imgs)))


@pytest.fixture()
def yolo_labels():
    def _read(label_path: str):
        rows = []
        if not os.path.exists(label_path):
            return rows
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                    rows.append((cls_id, cx, cy, w, h))
        return rows

    return _read


@pytest.fixture()
def to_label_path():
    def _fn(img_path: str) -> str:
        return _img_to_label(img_path)

    return _fn
