"""
Pruebas unitarias para la etapa `test` del pipeline de CI/CD.

Requisitos de la consigna:
  1. Probar que el modelo responde con datos de entrada definidos.
  2. Probar que una métrica (accuracy) no cae por debajo de un umbral.

El modelo ONNX y el dataset de prueba se descargan del bucket antes de
ejecutar estas pruebas (scripts/download_test_data.py). En local, se usan
los archivos generados por scripts/train_export_onnx.py.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import model  # noqa: E402

TEST_DATA_PATH = os.path.join("data", "iris_test.csv")


def _load_test_data():
    if not os.path.exists(TEST_DATA_PATH):
        pytest.skip(f"No existe {TEST_DATA_PATH}; ejecuta train_export_onnx.py o el CI.")
    raw = np.genfromtxt(TEST_DATA_PATH, delimiter=",", skip_header=1)
    X, y = raw[:, :-1], raw[:, -1].astype(int)
    return X, y


def test_modelo_responde_con_entrada_definida():
    """Prueba 1: el modelo devuelve una clase válida para un input conocido."""
    # Una setosa típica -> debería clasificarse como clase 0
    result = model.predict([5.1, 3.5, 1.4, 0.2])
    assert "label" in result and "class_name" in result
    assert result["label"] in (0, 1, 2)
    assert result["class_name"] in model.IRIS_CLASSES


def test_accuracy_supera_umbral():
    """Prueba 2: el accuracy sobre el set de prueba no cae por debajo del umbral."""
    threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))
    X, y_true = _load_test_data()
    y_pred = model.predict_batch(X)
    accuracy = float(np.mean(y_pred == y_true))
    print(f"accuracy={accuracy:.4f} (umbral={threshold})")
    assert accuracy >= threshold, (
        f"El accuracy {accuracy:.4f} cayó por debajo del umbral {threshold}."
    )
