import os
from pathlib import Path

# Carpeta donde estan los .joblib y model_cards
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts")).resolve()

# Nombre del modelo a cargar al iniciar
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "")  # si vacía: busca el .joblib más reciente

# Para activar logits/probabilidades en /predict
RETURN_PROBAS = os.getenv("RETURN_PROBAS", "1") == "1"