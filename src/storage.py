"""
Utilidades de acceso al bucket GCS: descarga del modelo/datos y
escritura (append) de los logs de predicciones.

Diseñado para funcionar en dos modos:
- Cloud (GCS): si hay credenciales y GCS_BUCKET configurado.
- Local (fallback): usa archivos del sistema de archivos para poder
  desarrollar y testear sin conexión a la nube.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

try:
    from google.cloud import storage  # type: ignore
    _GCS_AVAILABLE = True
except ImportError:  # pragma: no cover - entorno sin la librería
    _GCS_AVAILABLE = False


def _bucket():
    """Devuelve el bucket GCS o None si no está configurado/disponible."""
    bucket_name = os.getenv("GCS_BUCKET")
    if not bucket_name or not _GCS_AVAILABLE:
        return None
    try:
        client = storage.Client()
        return client.bucket(bucket_name)
    except Exception as exc:  # credenciales ausentes en local
        print(f"[storage] GCS no disponible ({exc}); usando modo local.")
        return None


def download_blob(blob_path: str, dest_path: str) -> str:
    """
    Descarga un objeto del bucket a una ruta local.
    En modo local, si ya existe el archivo destino, lo reutiliza.
    """
    bucket = _bucket()
    if bucket is None:
        if os.path.exists(dest_path):
            print(f"[storage] modo local: usando {dest_path} existente.")
            return dest_path
        raise FileNotFoundError(
            f"No hay bucket GCS configurado y no existe {dest_path} localmente. "
            f"Configura GCS_BUCKET o coloca el archivo en {dest_path}."
        )
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    bucket.blob(blob_path).download_to_filename(dest_path)
    print(f"[storage] descargado gs://{bucket.name}/{blob_path} -> {dest_path}")
    return dest_path


def append_prediction_log(line: str) -> None:
    """
    Agrega una línea al archivo de log de predicciones del entorno actual
    (predicciones_dev.txt o predicciones_prod.txt) dentro del bucket.
    En modo local escribe al archivo en disco.
    """
    env = os.getenv("ENVIRONMENT", "dev").lower()
    log_name = (
        os.getenv("PRED_LOG_PROD", "predicciones_prod.txt")
        if env == "prod"
        else os.getenv("PRED_LOG_DEV", "predicciones_dev.txt")
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    full_line = f"{timestamp} | {line}\n"

    bucket = _bucket()
    if bucket is None:
        with open(log_name, "a", encoding="utf-8") as fh:
            fh.write(full_line)
        return

    # GCS no soporta append nativo: leer-concatenar-reescribir.
    blob = bucket.blob(log_name)
    existing = blob.download_as_text() if blob.exists() else ""
    blob.upload_from_string(existing + full_line, content_type="text/plain")
