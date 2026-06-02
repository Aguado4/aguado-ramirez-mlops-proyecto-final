"""
Carga del modelo ONNX y ejecución de inferencias con onnxruntime.

El modelo se descarga del bucket GCS la primera vez que se necesita
(ver storage.download_blob). El archivo .onnx nunca vive en el repo.
"""
from __future__ import annotations

import os
import threading
from typing import List

import numpy as np
import onnxruntime as ort

from . import storage

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]

_session: ort.InferenceSession | None = None
_lock = threading.Lock()


def _local_model_path() -> str:
    return os.getenv("LOCAL_MODEL_PATH", "iris.onnx")


def get_session() -> ort.InferenceSession:
    """Devuelve una sesión ONNX cacheada (descarga el modelo si hace falta)."""
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                model_blob = os.getenv("MODEL_BLOB", "models/iris.onnx")
                dest = _local_model_path()
                storage.download_blob(model_blob, dest)
                _session = ort.InferenceSession(
                    dest, providers=["CPUExecutionProvider"]
                )
    return _session


def predict(features: List[float]) -> dict:
    """
    Ejecuta una predicción sobre un vector de features.
    Devuelve la clase predicha (índice y nombre).
    """
    session = get_session()
    x = np.array([features], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})
    label = int(np.ravel(outputs[0])[0])
    return {
        "label": label,
        "class_name": IRIS_CLASSES[label] if 0 <= label < len(IRIS_CLASSES) else str(label),
    }


def predict_batch(matrix: np.ndarray) -> np.ndarray:
    """Predicción vectorizada usada por la etapa de test (métrica de accuracy)."""
    session = get_session()
    x = matrix.astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})
    return np.ravel(outputs[0]).astype(int)
