# Customer Scoring API

API simple de FastAPI para scoring de clientes usando Machine Learning.

## 🚀 Características

- **FastAPI**: Framework moderno y rápido
- **Modelo ML**: Random Forest para predicción de scores
- **Gestión de dependencias**: UV (gestor de paquetes rápido)
- **Validación**: Pydantic para validación de datos
- **Documentación automática**: Swagger UI integrado

## 📋 Requisitos

- Python 3.11+
- UV (gestor de paquetes)

## 🔧 Instalación

1. Instalar dependencias:
```bash
uv sync
```

## 🎯 Uso

### 1. Entrenar el modelo

Primero, entrena el modelo de ML:

```bash
uv run python models/train_model.py
```

### 2. Iniciar el servidor

```bash
uv run uvicorn app.main:app --reload
```

El servidor estará disponible en: `http://localhost:8000`

### 3. Acceder a la documentación

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔍 Endpoints

### GET /
Información básica de la API

### GET /health
Verifica el estado de salud de la API

### POST /predict
Predice el score de un cliente

**Ejemplo de request:**
```json
{
  "customer_id": 12345
}
```

**Ejemplo de response:**
```json
{
  "customer_id": 12345,
  "score": 75.32,
  "score_normalized": 75.32,
  "features": {
    "segment": 45,
    "seniority": 28,
    "monthly_transactions": 5,
    "average_amount": 456.78
  }
}
```

### GET /predict/{customer_id}
Versión GET del endpoint de predicción

```bash
curl http://localhost:8000/predict/12345
```

## 🧪 Probar la API

### Usando curl:

```bash
# Health check
curl http://localhost:8000/health

# Predicción (POST)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 12345}'

# Predicción (GET)
curl http://localhost:8000/predict/12345
```

### Usando httpie:

```bash
http POST http://localhost:8000/predict customer_id:=12345
```

### Usando Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"customer_id": 12345}
)
print(response.json())
```

## 📁 Estructura del Proyecto

```
rk-input-data-scoring/
├── app/
│   ├── __init__.py
│   └── main.py          # API FastAPI
├── models/
│   ├── __init__.py
│   ├── train_model.py   # Entrenamiento del modelo
│   └── ml_model.joblib  # Modelo entrenado (se genera)
├── pyproject.toml       # Configuración del proyecto
├── uv.lock             # Lock file de dependencias
└── README.md
```

## 🛠️ Desarrollo

### Actualizar dependencias:
```bash
uv add <paquete>
```

### Ejecutar en modo desarrollo:
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📝 Notas

- El modelo usa datos sintéticos para demostración
- Las características se extraen del `customer_id`
- El score se normaliza entre 0 y 100
- Para producción, reemplaza los datos sintéticos con datos reales

## 🤝 Contribuir

Si quieres mejorar el modelo o agregar funcionalidades, sigue estos pasos:

1. Modifica `models/train_model.py` para mejorar el modelo
2. Re-entrena: `uv run python models/train_model.py`
3. Prueba los cambios en la API

## 📄 Licencia

MIT
