"""
API FastAPI que expone el modelo ONNX de clasificación Iris.

Cada predicción se registra (append) en el archivo de log del entorno
(predicciones_dev.txt / predicciones_prod.txt) dentro del bucket GCS,
para monitoreo y análisis posterior.
"""
from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import model, storage

app = FastAPI(
    title="MLOps Proyecto — Clasificador Iris (ONNX)",
    description="Despliegue automatizado dev/prod con modelo ONNX servido desde un bucket.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="Vector de 4 features: [sepal_length, sepal_width, petal_length, petal_width]",
        examples=[[5.1, 3.5, 1.4, 0.2]],
    )


class PredictResponse(BaseModel):
    label: int
    class_name: str
    environment: str


@app.get("/")
def root():
    return {
        "service": "iris-onnx",
        "environment": os.getenv("ENVIRONMENT", "dev"),
        "status": "ok",
    }


@app.get("/health")
def health():
    """Usado por el smoke test del pipeline CD."""
    try:
        model.get_session()
        return {"status": "test"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"modelo no disponible: {exc}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != 4:
        raise HTTPException(
            status_code=422, detail="Se esperan exactamente 4 features."
        )
    result = model.predict(req.features)
    env = os.getenv("ENVIRONMENT", "dev")

    # Registrar la predicción en el log del entorno (bucket)
    storage.append_prediction_log(
        f"env={env} features={req.features} -> "
        f"label={result['label']} class={result['class_name']}"
    )

    return PredictResponse(
        label=result["label"],
        class_name=result["class_name"],
        environment=env,
    )
