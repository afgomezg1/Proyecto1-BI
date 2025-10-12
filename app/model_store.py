import json, time, shutil, threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import joblib
import pandas as pd

from .config import ARTIFACTS_DIR, MODEL_FILENAME

_lock = threading.Lock()
_model = None
_model_path: Optional[Path] = None
_meta: Dict[str, Any] = {}

def _latest_joblib() -> Optional[Path]:
    jobs = sorted(ARTIFACTS_DIR.glob("*.joblib"))
    return jobs[-1] if jobs else None

def load_model() -> Tuple[Any, Path, Dict]:
    global _model, _model_path, _meta
    if _model is not None:
        return _model, _model_path, _meta

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / MODEL_FILENAME if MODEL_FILENAME else _latest_joblib()
    if not path or not path.exists():
        raise FileNotFoundError("No se encontró modelo .joblib en 'artifacts/'. Sube uno primero.")

    _model = joblib.load(path)
    _model_path = path

    # intenta cargar su card asociada (mismo timestamp si aplica)
    candidates = list(ARTIFACTS_DIR.glob("model_card_*.json"))
    _meta = {}
    if candidates:
        # toma la más reciente
        card = sorted(candidates)[-1]
        try:
            _meta = json.loads(card.read_text(encoding="utf-8"))
        except Exception:
            _meta = {}

    return _model, _model_path, _meta

def predict(texts_series: pd.Series, return_probas: bool = False):
    model, _, _ = load_model()
    y_pred = model.predict(texts_series)
    proba = model.predict_proba(texts_series) if (return_probas and hasattr(model, "predict_proba")) else None
    return y_pred, proba

def retrain_and_save(X_series: pd.Series, y, base_name: str = "nb_pipeline_v2") -> Path:
    """
    Clona el pipeline actual, hace fit con los nuevos datos y guarda a disco con timestamp.
    Retorna la ruta del nuevo .joblib.
    """
    from sklearn.base import clone
    import datetime

    with _lock:
        model, _, _ = load_model()
        new_model = clone(model)  # misma arquitectura/hiperparámetros
        new_model.fit(X_series, y)

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ARTIFACTS_DIR / f"{base_name}_{stamp}.joblib"

        # guardado atómico
        tmp = out_path.with_suffix(".tmp")
        joblib.dump(new_model, tmp)
        tmp.replace(out_path)

        # opcional: apunta/actualiza un alias "latest.joblib"
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

        return out_path
