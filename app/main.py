# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from .schemas import PredictRequest, PredictResponse, RetrainRequest
from .model_store import load_model, predict, retrain_incremental_and_save
from .config import RETURN_PROBAS

app = FastAPI(title="ODS Classifier API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.on_event("startup")
def _startup():
    load_model()

def _to_series(texts) -> pd.Series:
    return pd.Series(texts).fillna("").astype(str)

@app.get("/health")
def health():
    _, path, _ = load_model()
    return {"status": "ok", "model_path": str(path)}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    if not req.textos:
        raise HTTPException(status_code=400, detail="Lista de textos vacía")

    X_series = _to_series(req.textos)
    y_pred, proba = predict(X_series, return_probas=RETURN_PROBAS)
    preds = [str(p) for p in y_pred]

    probs_out = None
    if proba is not None:
        model, _, _ = load_model()
        classes = [str(c) for c in getattr(model, "classes_", [])]
        probs_out = [{cls: float(p) for cls, p in zip(classes, row)} for row in proba]

    return PredictResponse(predictions=preds, probabilities=probs_out)

@app.post("/retrain")
def retrain_endpoint(req: RetrainRequest):
    if len(req.textos) != len(req.labels):
        raise HTTPException(status_code=400, detail="textos y labels deben tener misma longitud")

    X_series = _to_series(req.textos)
    y = pd.Series(req.labels)

    try:
        path, before, after, sizes = retrain_incremental_and_save(X_series, y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reentrenando: {e}")

    return {
        "status": "ok",
        "new_model_path": str(path),
        "sizes": sizes,
        "metrics_before": before,   # accuracy, balanced_acc, precision_macro, recall_macro, f1_macro
        "metrics_after": after
    }
