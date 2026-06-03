# MLOps — Sistema de Despliegue Automático (Proyecto Final)

Sistema de **despliegue continuo de modelos ML** sobre la nube. Cada vez que se
publica una nueva versión del modelo o del código, el pipeline de CI/CD ejecuta
pruebas y, si pasan, despliega automáticamente un contenedor a un endpoint vivo.

El modelo de ejemplo es un clasificador **Iris** (RandomForest) exportado a
**ONNX**, servido mediante una API **FastAPI** y desplegado en **Google Cloud Run**.

---

## Arquitectura

```
GitHub (ramas dev / prod)
        │ push
        ▼
GitHub Actions ──► [test] descarga modelo+datos del bucket, corre pytest
        │                                  │ pasa
        ▼                                  ▼
   Artifact Registry  ◄── [build/promote] build Docker + deploy
        │
        ▼
   Cloud Run  ──► endpoint dev   (servicio iris-onnx-dev)
              ──► endpoint prod  (servicio iris-onnx-prod)
        │
        ▼
   GCS bucket: modelo .onnx, datos de prueba,
               predicciones_dev.txt / predicciones_prod.txt
```

- El archivo `.onnx` **no vive en el repo**: se referencia por variable de
  entorno (`MODEL_BLOB`) y se descarga del bucket en CI/CD y en runtime.
- Los **datos de prueba** tampoco están en el repo: se descargan del bucket en
  la etapa de test.
- Cada predicción se registra (append) en `predicciones_<env>.txt` en el bucket.

---

## Requisitos de la consigna — checklist

| Requisito | Dónde se cumple |
|---|---|
| Repo con CI/CD en GitHub Actions | `.github/workflows/dev.yml`, `prod.yml` |
| Ramas `dev` y `prod`, cada una con su endpoint | servicios `iris-onnx-dev` / `iris-onnx-prod` |
| Pipeline corre en cada push a dev/prod | `on: push: branches: [dev]` / `[prod]` |
| Modelo ONNX, no en el repo, referenciado en bucket | `MODEL_BLOB`, `.gitignore` ignora `*.onnx` |
| Etapa `test`: datos desde bucket + 2 pruebas | `scripts/download_test_data.py`, `tests/test_model.py` |
| Prueba 1: el modelo responde a entrada definida | `test_modelo_responde_con_entrada_definida` |
| Prueba 2: métrica sobre umbral | `test_accuracy_supera_umbral` |
| Etapa `build/promote`: Docker + app + deploy | `Dockerfile`, `src/app.py`, workflows |
| App de interacción (FastAPI) | `src/app.py` (`/predict`) |
| Logs de predicciones en bucket (2 archivos txt) | `src/storage.py::append_prediction_log` |

---

## Estructura

```
.
├── .github/workflows/
│   ├── dev.yml              # pipeline para la rama dev
│   └── prod.yml             # pipeline para la rama prod
├── src/
│   ├── app.py               # API FastAPI (/predict, /health)
│   ├── model.py             # carga ONNX + inferencia
│   └── storage.py           # descarga de bucket + log de predicciones
├── scripts/
│   ├── train_export_onnx.py # genera iris.onnx + datos de prueba (uso local)
│   └── download_test_data.py# descarga modelo + datos del bucket (CI)
├── tests/
│   └── test_model.py        # 2 pruebas unitarias
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Desarrollo local

Todo se puede probar localmente **sin nube** (modo fallback en `src/storage.py`).

```bash
# 1. entorno
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. generar modelo ONNX + datos de prueba
python scripts/train_export_onnx.py

# 3. correr las pruebas
pytest -v -s tests/

# 4. levantar la API
uvicorn src.app:app --reload --port 8080
# probar:
curl -X POST localhost:8080/predict -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---

## Puesta en marcha en la nube (pendiente)

1. **Crear bucket GCS** y subir el modelo y los datos una sola vez:
   ```bash
   gcloud storage buckets create gs://mlops-proyecto-artifacts --location=us-central1
   gcloud storage cp iris.onnx        gs://mlops-proyecto-artifacts/models/iris.onnx
   gcloud storage cp data/iris_test.csv gs://mlops-proyecto-artifacts/data/iris_test.csv
   ```
2. **Artifact Registry**: `gcloud artifacts repositories create mlops --repository-format=docker --location=us-central1`
3. **Service Account** con permisos (Cloud Run Admin, Storage Admin, Artifact Registry Writer) → exportar key JSON.
4. En GitHub → Settings → Secrets and variables → Actions:
   - **Secret** `GCP_SA_KEY` = contenido del JSON de la service account.
   - **Variables** `GCP_PROJECT`, `GCP_REGION` (ej. `us-central1`), `GCS_BUCKET`.
5. `git push` a `dev` o `prod` dispara el pipeline correspondiente.

---

## Endpoints en producción

| Entorno | URL |
|---|---|
| **dev** | https://iris-onnx-dev-4iuyx3z72q-uc.a.run.app |
| **prod** | https://iris-onnx-prod-4iuyx3z72q-uc.a.run.app |

Ejemplo de uso:
```bash
curl -X POST https://iris-onnx-dev-4iuyx3z72q-uc.a.run.app/predict \
     -H "Content-Type: application/json" -d '{"features":[5.1,3.5,1.4,0.2]}'
```

## Infraestructura GCP

| Recurso | Valor |
|---|---|
| Proyecto | `project-db6add9e-2647-4d19-892` (región `us-central1`) |
| Bucket | `gs://project-db6add9e-2647-4d19-892-mlops-artifacts` (modelo, datos, logs) |
| Artifact Registry | `us-central1-docker.pkg.dev/.../mlops/iris-onnx` |
| Auth CI/CD | **Workload Identity Federation** (keyless) — la org bloquea SA keys |
| Protección de costos | **Kill-switch**: budget → Pub/Sub → Cloud Function que desvincula billing |

## Estado actual

- [x] Estructura del repo, ramas `dev`/`prod`
- [x] Modelo ONNX (script de generación) + app FastAPI + logging
- [x] Pruebas unitarias + scripts de descarga
- [x] Dockerfile + workflows de CI/CD
- [x] Bucket GCS creado y artefactos subidos
- [x] Auth keyless con Workload Identity Federation (sin SA keys, por política de org)
- [x] Deploy real a Cloud Run dev y prod (ambos endpoints vivos)
- [x] Kill-switch de facturación montado y verificado
- [ ] Demo de sustentación (10 min)
