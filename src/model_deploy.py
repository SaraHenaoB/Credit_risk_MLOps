# comando para ejecutar la API: uvicorn model_deploy:app --reload

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"

app = FastAPI(
    title="API de predicción de riesgo crediticio",
    description=(
        "API para predecir la clase de pago de uno o varios clientes "
        "utilizando el modelo entrenado y su preprocesador."
    ),
    version="1.2.0",
)


# --------------------------------------------------
# CARGA DEL MODELO Y DEL PREPROCESADOR
# --------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Variables registradas durante el entrenamiento.
    EXPECTED_FEATURES = list(preprocessor.feature_names_in_)

    print("Modelo cargado exitosamente.")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Preprocesador: {PREPROCESSOR_PATH}")
    print(f"Cantidad de variables esperadas: {len(EXPECTED_FEATURES)}")
    print(f"Clases del modelo: {model.classes_}")

except Exception as e:
    model = None
    preprocessor = None
    EXPECTED_FEATURES = []

    print(f"Error al cargar el modelo o preprocesador: {e}")


# --------------------------------------------------
# ESTRUCTURA DE ENTRADA
# --------------------------------------------------

class BatchInput(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)


# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "API de predicción de riesgo crediticio activa",
        "docs": "/docs",
        "endpoint_predictions": "/predict",
    }


@app.get("/saludo")
def saludo():
    return {
        "message": "La API está funcionando correctamente."
    }


@app.post("/predict")
def predict_batch(input_data: BatchInput):

    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo o el preprocesador no pudieron cargarse.",
        )

    try:
        records = input_data.records

        # Validar las variables de cada registro.
        for i, record in enumerate(records, start=1):
            missing = [
                col for col in EXPECTED_FEATURES
                if col not in record
            ]

            extra = [
                col for col in record
                if col not in EXPECTED_FEATURES
            ]

            if missing:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "registro": i,
                        "mensaje": "Faltan variables obligatorias.",
                        "variables_faltantes": missing,
                    },
                )

            if extra:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "registro": i,
                        "mensaje": "Hay variables no esperadas.",
                        "variables_adicionales": extra,
                    },
                )

        # Mantener el mismo orden de columnas del entrenamiento.
        df = pd.DataFrame(records).reindex(
            columns=EXPECTED_FEATURES
        )

        # Replicar la limpieza aplicada durante el entrenamiento.
        if "tendencia_ingresos" in df.columns:
            valid_values = [
                "Estable",
                "Creciente",
                "Decreciente",
            ]

            df["tendencia_ingresos"] = df[
                "tendencia_ingresos"
            ].apply(
                lambda value: (
                    value if value in valid_values else None
                )
            )

        # Aplicar el preprocesamiento guardado.
        X_processed = preprocessor.transform(df)

        # Generar predicciones.
        predictions = model.predict(X_processed)

        # Obtener probabilidades cuando el modelo las soporte.
        probabilities = (
            model.predict_proba(X_processed)
            if hasattr(model, "predict_proba")
            else None
        )

        classes = list(model.classes_)
        results = []

        for i, prediction in enumerate(predictions):

            predicted_class = prediction.item() if hasattr(
                prediction, "item"
            ) else prediction

            class_probabilities = {}

            if probabilities is not None:
                for j, class_label in enumerate(classes):
                    label = (
                        class_label.item()
                        if hasattr(class_label, "item")
                        else class_label
                    )

                    class_probabilities[str(label)] = float(
                        probabilities[i, j]
                    )
                    
            predicted_probability = class_probabilities.get(
                str(predicted_class)
            )

            if predicted_class == 1:
                interpretation = "Paga a tiempo"
            elif predicted_class == 0:
                interpretation = "Incumplimiento"
            else:
                interpretation = "Clase no reconocida"

            results.append({
                "registro": i + 1,
                "clase_predicha": int(predicted_class),
                "interpretacion": interpretation,
                "probabilidad_por_clase": class_probabilities,
                "confianza_clase_predicha": predicted_probability,
                "probabilidad_pago_a_tiempo": class_probabilities.get("1"),
                "probabilidad_incumplimiento": class_probabilities.get("0"),
            })

        return {
            "cantidad_registros": len(results),
            "predicciones": results,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la predicción: {str(e)}",
        )
