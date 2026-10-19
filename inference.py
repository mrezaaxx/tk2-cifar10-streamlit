"""Logika inferensi bersama untuk app.py dan evaluate_new_images.py."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent


def load_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def load_model(config):
    import tensorflow as tf  # impor lambat agar error versi tertangkap pemanggil

    path = ROOT / config["model_path"]
    if not path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")
    return tf.keras.models.load_model(path)


def resize_for_model(image: Image.Image, config) -> Image.Image:
    """RGB -> resize ke ukuran input model dengan metode resample dari config."""
    resample = Image.Resampling[config.get("resample", "bilinear").upper()]
    return image.convert("RGB").resize(tuple(config["input_size"]), resample)


def preprocess(image: Image.Image, config) -> np.ndarray:
    """RGB -> resize ke input model -> normalisasi -> tambah dimensi batch."""
    arr = np.asarray(resize_for_model(image, config), dtype="float32")
    if config["normalization"] == "rescale_255":
        arr = arr / 255.0
    else:
        raise ValueError(f"Normalisasi tidak dikenal: {config['normalization']}")
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image, config, top_k=3):
    """Kembalikan list (nama_kelas, probabilitas) terurut menurun sepanjang top_k."""
    probs = model.predict(preprocess(image, config), verbose=0)[0]
    order = np.argsort(probs)[::-1][:top_k]
    return [(config["class_names"][i], float(probs[i])) for i in order]
