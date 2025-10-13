# app/model_store.py
import json, shutil, threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)

from sklearn.model_selection import train_test_split

from .config import (
    ARTIFACTS_DIR, MODEL_FILENAME,
    DATASTORE_FILENAME, RETRAIN_EVAL_ON, DEDUP_HISTORY,
    EVALSTORE_FILENAME, HOLDOUT_SIZE, HOLDOUT_MIN_SAMPLES
)
from .text_normalizer import TextNormalizer, process_text_once

# Compat c/ modelos guardados desde notebooks
import sys as _sys
try:
    _sys.modules['__main__'].TextNormalizer = TextNormalizer
    _sys.modules['__main__'].process_text_once = process_text_once
except Exception:
    pass

_lock = threading.Lock()
_model = None
_model_path: Optional[Path] = None
_meta: Dict[str, Any] = {}
_model_mtime: Optional[float] = None

def _latest_joblib() -> Optional[Path]:
    jobs = sorted(ARTIFACTS_DIR.glob("*.joblib"))
    return jobs[-1] if jobs else None

def _load_from_path(path: Path):
    """Carga el modelo desde 'path' y actualiza cache global + meta."""
    global _model, _model_path, _model_mtime, _meta
    _model = joblib.load(path)
    _model_path = path
    try:
        _model_mtime = path.stat().st_mtime
    except Exception:
        _model_mtime = None

    # recarga card/metadata (si existe)
    cards = list(ARTIFACTS_DIR.glob("model_card_*.json"))
    _meta = {}
    if cards:
        try:
            _meta = json.loads(sorted(cards)[-1].read_text(encoding="utf-8"))
        except Exception:
            _meta = {}


def load_model() -> Tuple[Any, Path, Dict]:
    """
    Devuelve el modelo cacheado; si detecta que en disco hay un .joblib más nuevo
    (por nombre o por mtime cuando se usa 'latest.joblib'), refresca el cache.
    """
    global _model, _model_path, _model_mtime, _meta

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ¿Qué archivo deberíamos tener cargado?
    desired = ARTIFACTS_DIR / MODEL_FILENAME if MODEL_FILENAME else _latest_joblib()
    if not desired or not desired.exists():
        raise FileNotFoundError("No se encontró modelo .joblib en 'artifacts/'. Sube uno primero.")

    # Si no hay modelo en cache, carga directo
    if _model is None or _model_path is None:
        _load_from_path(desired)
        return _model, _model_path, _meta

    # Si la ruta cambió (nuevo timestamp) => recarga
    try:
        if desired.resolve() != _model_path.resolve():
            _load_from_path(desired)
            return _model, _model_path, _meta
    except Exception:
        # por si resolve() falla en alguna plataforma
        if str(desired) != str(_model_path):
            _load_from_path(desired)
            return _model, _model_path, _meta

    # Si la ruta es la misma (p.ej. latest.joblib) pero el archivo fue reemplazado => recarga por mtime
    try:
        current_mtime = desired.stat().st_mtime
        if _model_mtime is None or current_mtime != _model_mtime:
            _load_from_path(desired)
    except Exception:
        # si no se puede leer mtime, seguimos con el cache actual
        pass

    return _model, _model_path, _meta

def predict(texts_series: pd.Series, return_probas: bool = False):
    model, _, _ = load_model()
    y_pred = model.predict(texts_series)
    proba = model.predict_proba(texts_series) if (return_probas and hasattr(model, "predict_proba")) else None
    return y_pred, proba

# ------------------------------- #
#   Persistencia de HISTÓRICO     #
# ------------------------------- #
def _history_path() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR / DATASTORE_FILENAME

def _load_history_df() -> pd.DataFrame:
    p = _history_path()
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame(columns=["texto", "label"])

def _save_history_df(df: pd.DataFrame) -> None:
    p = _history_path()
    df.to_csv(p, index=False, encoding="utf-8")

def _to_series(x) -> pd.Series:
    return pd.Series(x).fillna("").astype(str)

def _metrics(model, X: pd.Series, y: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(_to_series(X))
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
    }

# ------------------------------- #
#   Eval set fijo (holdout)       #
# ------------------------------- #

def _eval_path() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR / EVALSTORE_FILENAME

def _load_eval_df() -> pd.DataFrame:
    p = _eval_path()
    if p.exists():
        try:
            df = pd.read_csv(p)
            df["texto"] = df["texto"].astype(str)
            df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["label"]).astype({"label": int})
            return df[["texto", "label"]]
        except Exception:
            pass
    return pd.DataFrame(columns=["texto", "label"])

def _save_eval_df(df: pd.DataFrame) -> None:
    p = _eval_path()
    df.to_csv(p, index=False, encoding="utf-8")

# ------------------------------------------ #
#   Reentrenamiento incremental + métricas   #
# ------------------------------------------ #
def retrain_incremental_and_save(
    X_new: pd.Series, y_new,
    base_name: str = "nb_pipeline_v2",
    eval_on: str = RETRAIN_EVAL_ON
) -> Tuple[Path, Dict[str, float], Dict[str, float], Dict[str, int]]:
    """
    Reentrena ACUMULANDO histórico y evalúa antes/después:
    - Si existe artifacts/eval_holdout.csv => evalúa SIEMPRE allí (y excluye ese holdout del entrenamiento).
    - Si no existe y hay histórico suficiente => crea holdout estratificado una sola vez y lo guarda.
    - Si no hay histórico suficiente y no hay holdout => respeta eval_on ("new" o "combined").
    Retorna: (ruta_modelo, metrics_before, metrics_after, sizes_info)
    """
    import datetime

    with _lock:
        model, _, _ = load_model()

        # --- cargar histórico y lote nuevo
        hist_df = _load_history_df()
        hist_df["texto"] = hist_df["texto"].astype(str)
        try:
            hist_df["label"] = pd.to_numeric(hist_df["label"], errors="raise").astype(int)
        except Exception:
            pass

        X_new = _to_series(X_new)
        y_new = pd.Series(y_new)

        # --- cargar/crear eval fijo
        eval_df = _load_eval_df()
        use_fixed_eval = len(eval_df) > 0

        # si no hay eval fijo pero tenemos suficiente histórico, lo creamos una sola vez
        if not use_fixed_eval and len(hist_df) >= HOLDOUT_MIN_SAMPLES:
            X_tr, X_ev, y_tr, y_ev = train_test_split(
                hist_df["texto"], hist_df["label"],
                test_size=HOLDOUT_SIZE, random_state=42, stratify=hist_df["label"]
            )
            eval_df = pd.DataFrame({"texto": X_ev.astype(str), "label": y_ev.astype(int)})
            _save_eval_df(eval_df)   # persistimos para próximas llamadas
            # y el histórico que queda para entrenar es X_tr/y_tr
            hist_df = pd.DataFrame({"texto": X_tr.astype(str), "label": y_tr.astype(int)})
            use_fixed_eval = True

        # --- preparar set de evaluación
        if use_fixed_eval:
            # prioridad: eval fijo; NO debe entrar al entrenamiento
            eval_X = eval_df["texto"]
            eval_y = eval_df["label"]
            # más abajo vamos a excluir eval_df del train (por texto) para evitar fuga
            eval_text_set = set(eval_X.tolist())
        else:
            # fallback: tu comportamiento previo
            if eval_on == "combined" and len(hist_df) > 0:
                eval_X = pd.concat([hist_df["texto"], X_new], ignore_index=True)
                eval_y = pd.concat([hist_df["label"], y_new], ignore_index=True)
            else:
                eval_X, eval_y = X_new, y_new
            eval_text_set = set()  # no excluimos nada

        # --- métricas ANTES con el modelo actual
        metrics_before = _metrics(model, eval_X, eval_y)

        # --- acumular + deduplicar histórico
        new_df = pd.DataFrame({"texto": X_new, "label": y_new})
        all_df = pd.concat([hist_df, new_df], ignore_index=True)

        if DEDUP_HISTORY:
            all_df = all_df.drop_duplicates(subset=["texto", "label"]).reset_index(drop=True)

        # --- excluir del entrenamiento el eval fijo (si aplica)
        if use_fixed_eval and len(eval_text_set) > 0:
            mask = ~all_df["texto"].isin(eval_text_set)
            train_df = all_df.loc[mask].reset_index(drop=True)
        else:
            train_df = all_df

        # --- entrenar clon con el TRAIN (sin eval fijo)
        new_model = clone(model)
        new_model.fit(_to_series(train_df["texto"]), pd.Series(train_df["label"]))

        # --- métricas DESPUÉS sobre el MISMO eval set
        metrics_after = _metrics(new_model, eval_X, eval_y)

        # --- guardar .joblib y puntero latest
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ARTIFACTS_DIR / f"{base_name}_{stamp}.joblib"
        tmp = out_path.with_suffix(".tmp")
        joblib.dump(new_model, tmp)
        tmp.replace(out_path)

        latest = ARTIFACTS_DIR / "latest.joblib"
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
        except Exception:
            pass
        shutil.copyfile(out_path, latest)

        # swap en memoria
        global _model, _model_path
        _model = new_model
        _model_path = out_path
        
        try:
            from os import stat
            global _model_mtime
            _model_mtime = out_path.stat().st_mtime
        except Exception:
            pass

        # --- persistir histórico actualizado
        _save_history_df(all_df)

        sizes = {
            "n_history_before": int(len(hist_df)),
            "n_new": int(len(new_df)),
            "n_history_after": int(len(all_df)),
        }
        return out_path, metrics_before, metrics_after, sizes
