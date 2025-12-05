"""
Script para entrenar un modelo simple de ML
Este modelo predice un score basado en el ID del cliente
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os


def create_synthetic_data(n_samples=1000):
    """
    Crea datos sintéticos para entrenamiento
    Simula características de clientes basadas en sus IDs
    """
    np.random.seed(42)

    # Generar IDs de clientes
    customer_ids = np.arange(1, n_samples + 1)

    # Crear características sintéticas basadas en el ID
    features = np.column_stack(
        [
            customer_ids % 100,  # Segmento
            (customer_ids * 7) % 50,  # Antigüedad simulada
            np.random.randint(0, 10, n_samples),  # Transacciones mensuales
            np.random.uniform(0, 1000, n_samples),  # Monto promedio
        ]
    )

    # Crear target (score) con alguna lógica
    scores = (
        features[:, 0] * 0.3
        + features[:, 1] * 0.2
        + features[:, 2] * 10
        + features[:, 3] * 0.05
        + np.random.normal(0, 10, n_samples)
    )

    # Normalizar scores entre 0 y 100
    scores = np.clip(scores, 0, 100)

    return features, scores


def train_and_save_model():
    """
    Entrena el modelo y lo guarda en disco
    """
    print("Generando datos sintéticos...")
    X, y = create_synthetic_data(n_samples=1000)

    print("Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Entrenando modelo Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluar modelo
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"R² Score - Train: {train_score:.4f}, Test: {test_score:.4f}")

    # Guardar modelo
    model_path = os.path.join(os.path.dirname(__file__), "ml_model.joblib")
    joblib.dump(model, model_path)
    print(f"Modelo guardado en: {model_path}")

    return model


if __name__ == "__main__":
    train_and_save_model()
