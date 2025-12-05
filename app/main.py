"""
API FastAPI para scoring de clientes
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from typing import Optional


# Cargar el modelo al iniciar la aplicación
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ml_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Modelo cargado exitosamente desde: {MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠ Modelo no encontrado en: {MODEL_PATH}")
    print("Por favor, ejecuta: uv run python models/train_model.py")
    model = None


# Crear aplicación FastAPI
app = FastAPI(
    title="Customer Scoring API",
    description="API para predecir scores de clientes basado en su ID",
    version="1.0.0",
)

# Configurar templates
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)


# Modelos de datos con Pydantic
class CustomerRequest(BaseModel):
    customer_id: int = Field(
        ..., gt=0, description="ID único del cliente (debe ser mayor a 0)"
    )

    class Config:
        json_schema_extra = {"example": {"customer_id": 12345}}


class ScoreResponse(BaseModel):
    customer_id: int
    score: float
    score_normalized: float
    features: dict

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": 12345,
                "score": 75.32,
                "score_normalized": 75.32,
                "features": {
                    "segment": 45,
                    "seniority": 28,
                    "monthly_transactions": 5,
                    "average_amount": 456.78,
                },
            }
        }


def extract_features(customer_id: int) -> np.ndarray:
    """
    Extrae características del customer_id
    Debe coincidir con la lógica usada en el entrenamiento
    """
    np.random.seed(customer_id)  # Para consistencia en características aleatorias

    features = np.array(
        [
            [
                customer_id % 100,  # Segmento
                (customer_id * 7) % 50,  # Antigüedad simulada
                np.random.randint(0, 10),  # Transacciones mensuales
                np.random.uniform(0, 1000),  # Monto promedio
            ]
        ]
    )

    return features


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """
    Página principal con interfaz HTML para visualizar inputs y hacer predicciones
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api")
def api_info():
    """
    Endpoint de información de la API (versión JSON)
    """
    return {
        "message": "Customer Scoring API",
        "version": "1.0.0",
        "status": "online" if model is not None else "waiting for model",
        "endpoints": {
            "home": "/",
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health_check():
    """
    Verifica el estado de salud de la API
    """
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=ScoreResponse)
def predict_score(request: CustomerRequest):
    """
    Predice el score para un cliente basado en su ID

    Args:
        request: Objeto con el customer_id

    Returns:
        ScoreResponse con el score predicho y las características
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta el entrenamiento primero.",
        )

    try:
        # Extraer características del customer_id
        features = extract_features(request.customer_id)

        # Realizar predicción
        score = float(model.predict(features)[0])

        # Normalizar score entre 0 y 100
        score_normalized = np.clip(score, 0, 100)

        # Preparar respuesta
        response = ScoreResponse(
            customer_id=request.customer_id,
            score=round(score, 2),
            score_normalized=round(score_normalized, 2),
            features={
                "segment": int(features[0][0]),
                "seniority": int(features[0][1]),
                "monthly_transactions": int(features[0][2]),
                "average_amount": round(float(features[0][3]), 2),
            },
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al procesar la predicción: {str(e)}"
        )


@app.get("/predict/{customer_id}")
def predict_score_get(customer_id: int):
    """
    Versión GET del endpoint de predicción (alternativa)

    Args:
        customer_id: ID del cliente en la URL

    Returns:
        Score predicho
    """
    if customer_id <= 0:
        raise HTTPException(status_code=400, detail="El customer_id debe ser mayor a 0")

    request = CustomerRequest(customer_id=customer_id)
    return predict_score(request)
