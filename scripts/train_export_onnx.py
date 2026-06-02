"""
Entrena un clasificador sencillo (Iris) y lo exporta a formato ONNX.

Este script es de uso LOCAL / preparación: genera el artefacto `iris.onnx`
y el dataset de prueba `iris_test.csv`. Ninguno de los dos se versiona en el
repo (ver .gitignore); deben subirse manualmente al bucket GCS una sola vez:

    gcloud storage cp iris.onnx       gs://$GCS_BUCKET/models/iris.onnx
    gcloud storage cp data/iris_test.csv gs://$GCS_BUCKET/data/iris_test.csv

En CI/CD el modelo y los datos se DESCARGAN del bucket, nunca del repo.
"""
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import to_onnx


def main() -> None:
    iris = load_iris()
    X, y = iris.data.astype(np.float32), iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Accuracy en test: {acc:.4f}")

    # Exportar a ONNX
    onx = to_onnx(clf, X_train[:1])
    with open("iris.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    print("Modelo exportado -> iris.onnx")

    # Guardar dataset de prueba (features + label) para la etapa de test del CI
    os.makedirs("data", exist_ok=True)
    test_path = os.path.join("data", "iris_test.csv")
    header = ",".join([f"f{i}" for i in range(X.shape[1])] + ["label"])
    rows = np.column_stack([X_test, y_test])
    np.savetxt(test_path, rows, delimiter=",", header=header,
               comments="", fmt=["%.6f"] * X.shape[1] + ["%d"])
    print(f"Datos de prueba exportados -> {test_path} ({len(X_test)} filas)")


if __name__ == "__main__":
    main()
