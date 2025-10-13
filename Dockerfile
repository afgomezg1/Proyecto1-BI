# Imagen base ligera con Python 3.12
FROM python:3.12-slim

# Ajustes de Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data

# curl para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Dependencias primero (mejor caché)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Crear explícitamente el directorio de NLTK antes de descargar datos
RUN mkdir -p /usr/local/share/nltk_data

# *** Hornear datos NLTK necesarios ***
RUN python - <<'PY'
import nltk
for p in ["punkt","punkt_tab","stopwords"]:
    nltk.download(p, quiet=True)
print("NLTK data preloaded")
PY

# Copiar SOLO el código de la API
COPY app/ app/

# Usuario no-root y permisos
RUN useradd -u 1000 -m appuser && \
    chown -R appuser:appuser /app /usr/local/share/nltk_data
USER appuser

# Variables del backend
ENV ARTIFACTS_DIR=/artifacts \
    MODEL_FILENAME="" \
    RETURN_PROBAS=1

# Puerto interno
EXPOSE 8000

# Healthcheck a /health
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Arranque (FastAPI en app/main.py) con Gunicorn + UvicornWorker
CMD ["gunicorn","-w","2","-k","uvicorn.workers.UvicornWorker","-b","0.0.0.0:8000","app.main:app"]
