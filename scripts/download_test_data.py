"""
Descarga los datos de prueba y el modelo ONNX desde el bucket GCS.

Lo usa la etapa `test` del pipeline de CI/CD. Ni el modelo ni los datos
existen en el repo: se obtienen aquí desde el bucket.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import storage  # noqa: E402


def main() -> None:
    model_blob = os.getenv("MODEL_BLOB", "models/iris.onnx")
    test_blob = os.getenv("TEST_DATA_BLOB", "data/iris_test.csv")

    storage.download_blob(model_blob, os.getenv("LOCAL_MODEL_PATH", "iris.onnx"))
    storage.download_blob(test_blob, "data/iris_test.csv")
    print("Modelo y datos de prueba descargados correctamente.")


if __name__ == "__main__":
    main()
