# Pasos para correr API

## 1) Activar venv y deps

```
pip install -r requirements.txt
```

## 2) Coloca modelo .joblib en artifacts/ (Ya debería de estar el modelo más reciente)

   └─ artifacts/nb_pipeline_v2_20251012_140030.joblib

## 3) Lanzar API

```
uvicorn app.main:app --reload --port 8000
```

# Probar rapidamente con curl:

Para predecir:

```
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"textos": ["Este proyecto busca mejorar la educación en zonas rurales."]}'
```

Para reentrenar:

```
curl -X POST "http://localhost:8000/retrain" \
  -H "Content-Type: application/json" \
  -d '{"textos": ["nuevo texto ...", "otro texto ..."], "labels": [4,3]}'

```

# Probar rápidamente con documentación de FastAPI:

Simplemente ingresar a la documentación y usar la interfaz gráfica:

```
http://127.0.0.1:8000/docs

```
