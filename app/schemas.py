from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    textos: List[str] = Field(..., description="Lista de textos en español (crudos)")

class PredictResponse(BaseModel):
    predictions: List[str]
    probabilities: Optional[List[Dict[str, float]]] = None  # por fila, dict de clase->prob

class RetrainRequest(BaseModel):
    textos: List[str]
    labels: List[int]  # o List[str] si etiquetas no son ints
