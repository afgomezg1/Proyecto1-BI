# app/config.py
import os
from pathlib import Path

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts")).resolve()
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "")
RETURN_PROBAS = os.getenv("RETURN_PROBAS", "1") == "1"

# === NUEVO: archivo donde persistimos el histórico de entrenamiento
DATASTORE_FILENAME = os.getenv("DATASTORE_FILENAME", "train_store.csv")

# Evaluación durante el retrain: "new" (sobre el lote nuevo) o "combined"
RETRAIN_EVAL_ON = os.getenv("RETRAIN_EVAL_ON", "new")
# ¿Eliminar duplicados por (texto, label) al acumular?
DEDUP_HISTORY = os.getenv("DEDUP_HISTORY", "1") == "1"

# (NUEVO) archivo con el holdout de evaluación “fijo”
EVALSTORE_FILENAME = os.getenv("EVALSTORE_FILENAME", "eval_holdout.csv")
# (NUEVO) tamaño de holdout si hay que crearlo desde el histórico (solo la primera vez)
HOLDOUT_SIZE = float(os.getenv("HOLDOUT_SIZE", "0.15"))
# (NUEVO) tamaño mínimo del histórico para poder crear holdout
HOLDOUT_MIN_SAMPLES = int(os.getenv("HOLDOUT_MIN_SAMPLES", "60"))