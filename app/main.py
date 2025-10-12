from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

from .schemas import PredictRequest, PredictResponse, RetrainRequest
from .preprocessing import preprocess_for_api
from .model_store import load_model, predict, retrain_and_save
from .config import RETURN_PROBAS

app = FastAPI(title="ODS Classifier API", version="2.0")

# CORS (ajustar orígenes según  UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.on_event("startup")
def _startup():
    # levanta el modelo al iniciar
    load_model()

@app.get("/health")
def health():
    _, path, _ = load_model()
    return {"status": "ok", "model_path": str(path)}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    if not req.textos:
        raise HTTPException(status_code=400, detail="Lista de textos vacía")

    # 1) Normalizar exactamente como en el entrenamiento
    X_norm = preprocess_for_api(req.textos, remove_stopwords=True)

    # 2) Predecir
    y_pred, proba = predict(X_norm, return_probas=RETURN_PROBAS)
    preds = [str(p) for p in y_pred]

    probs_out = None
    if proba is not None:
        model, _, _ = load_model()
        classes = [str(c) for c in getattr(model, "classes_", [])]
        probs_out = []
        for row in proba:
            probs_out.append({cls: float(p) for cls, p in zip(classes, row)})

    return PredictResponse(predictions=preds, probabilities=probs_out)

@app.post("/retrain")
def retrain_endpoint(req: RetrainRequest):
    if len(req.textos) != len(req.labels):
        raise HTTPException(status_code=400, detail="textos y labels deben tener misma longitud")

    # Normaliza textos crudos
    X_norm = preprocess_for_api(req.textos, remove_stopwords=True)
    y = pd.Series(req.labels)

    try:
        path = retrain_and_save(X_norm, y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reentrenando: {e}")

    return {"status": "ok", "new_model_path": str(path)}
