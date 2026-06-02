FROM python:3.11-slim

WORKDIR /app

# Dependencias de runtime (subconjunto liviano para el contenedor)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    onnxruntime==1.19.2 \
    numpy==1.26.4 \
    pydantic==2.9.2 \
    google-cloud-storage==2.18.2

COPY src/ ./src/

# El modelo ONNX NO se copia: se descarga del bucket al iniciar (src/model.py)
# Variables esperadas en runtime: GCS_BUCKET, MODEL_BLOB, ENVIRONMENT, etc.

EXPOSE 8080

# Cloud Run inyecta $PORT; por defecto 8080
ENV PORT=8080
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
