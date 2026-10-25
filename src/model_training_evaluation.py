# Importar librerías
import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score

from ft_engineering import ft_engineering_pipeline
from cargar_datos import cargarDatos

import warnings
warnings.filterwarnings('ignore')

def build_model(model_name):
    """
    Definición de modelos teniendo en cuenta el desbalanceo de datos
    """
    models = {
        "Logistic_Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=15,
                                                 class_weight="balanced_subsample", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5,learning_rate=0.05, scale_pos_weight=19,
                                                 random_state=42, eval_metric="logloss"), # Relación aprox 95/5
        "LightGBM": LGBMClassifier(n_estimators=100, max_depth=5, num_leaves=10, class_weight="balanced", 
                                                random_state=42, verbose=-1)
    }
    
    if model_name not in models:
        raise ValueError(f"Modelo {model_name} no está configurado en build_model.")
        
    return models[model_name]


def summarize_classification(model, X_train, X_test, y_train, y_test, model_name):
    """
    Entrena el modelo, mide tiempos (Escalabilidad), calcula predicciones 
    e itera sobre Train y Test para evaluar consistencia.
    """
    print(f"Entrenando y evaluando: {model_name}")
    
    # 1. Medición de Escalabilidad (Tiempo de entrenamiento)
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    # 2. Predicciones
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    probs_train = model.predict_proba(X_train)[:, 1]
    probs_test = model.predict_proba(X_test)[:, 1]
    
    # 3. Cálculo de métricas básicas (Enfocadas en desbalanceo)
    f1_train = f1_score(y_train, preds_train, average="macro")
    f1_test = f1_score(y_test, preds_test, average="macro")
    
    roc_train = roc_auc_score(y_train, probs_train)
    roc_test = roc_auc_score(y_test, probs_test)
    
    # 4. Evaluación de Consistencia (Brecha entre Train y Test para Overfitting)
    gap_f1 = abs(f1_train - f1_test)
    
    # Resumen en un diccionario para la tabla comparativa final
    summary = {
        "Model": model_name,
        "Train_F1_Macro": f1_train,
        "Test_F1_Macro": f1_test,
        "Train_ROC_AUC": roc_train,
        "Test_ROC_AUC": roc_test,
        "Consistency_Gap": gap_f1,
        "Training_Time_Sec": elapsed_time,
        "Object": model
    }
    
    return summary

def plot_comparisons(df_metrics):
    """
    Genera los gráficos comparativos para Performance, 
    Consistencia y Escalabilidad
    """
    plt.figure(figsize=(16, 5))
    
    # Gráfico 1: Performance (F1-Score en Test)
    plt.subplot(1, 3, 1)
    sns.barplot(x="Model", y="Test_F1_Macro", data=df_metrics, palette="viridis")
    plt.title("Performance: F1-Score Macro (Mayor es mejor)")
    plt.xticks(rotation=15)
    plt.ylabel("F1-Score Macro")
    
    # Gráfico 2: Consistency (Brecha Train vs Test)
    plt.subplot(1, 3, 2)
    sns.barplot(x="Model", y="Consistency_Gap", data=df_metrics, palette="magma")
    plt.title("Consistency: Brecha Train/Test (Menor es mejor)")
    plt.xticks(rotation=15)
    plt.ylabel("Diferencia de F1-Score")
    
    # Gráfico 3: Scalability (Tiempo de entrenamiento)
    plt.subplot(1, 3, 3)
    sns.barplot(x="Model", y="Training_Time_Sec", data=df_metrics, palette="coolwarm")
    plt.title("Scalability: Tiempo de Cómputo (Menor es mejor)")
    plt.xticks(rotation=15)
    plt.ylabel("Segundos")
    
    plt.tight_layout()
    plt.savefig(
        "metricas_comparativas_modelos.png",
        bbox_inches="tight",
        dpi=150
    )
    plt.close()
    print("\n[INFO] Gráfico comparativo guardado como 'metricas_comparativas_modelos.png'")

def main():

    print(" INICIANDO PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")

    
    # Carga e Ingeniería de características automática
    df_raw = cargarDatos()
    X_train, X_test, y_train, y_test, preprocessor = ft_engineering_pipeline(df_raw)
    
    # Listado de modelos a evaluar
    lista_modelos = ["Logistic_Regression", "Random_Forest", "XGBoost", "LightGBM"]
    all_summaries = []
    
    # Iteración y entrenamiento usando build_model y summarize_classification
    for name in lista_modelos:
        model_obj = build_model(name)
        summary = summarize_classification(model_obj, X_train, X_test, y_train, y_test, name)
        all_summaries.append(summary)
        
    # Tabla Resumen
    df_metrics = pd.DataFrame(all_summaries)

    # Selección automática del mejor modelo basado en el mayor Test F1-Macro
    best_row = df_metrics.loc[df_metrics["Test_F1_Macro"].idxmax()]
    best_model_object = best_row["Object"]
    
    print("\n==================================================================")
    print("                     TABLA RESUMEN DE EVALUACIÓN                  ")
    print("==================================================================")
    print(df_metrics.drop(columns=["Object"]).to_string(index=False))
    print("==================================================================\n")
    
    print(f"EL MODELO SELECCIONADO COMO EL MEJOR ES: {best_row['Model']}")
    print(f"   -> Performance (Test F1-Macro): {best_row['Test_F1_Macro']:.4f}")
    print(f"   -> Consistencia (Brecha Train/Test): {best_row['Consistency_Gap']:.4f}")
    print(f"   -> Escalabilidad (Tiempo): {best_row['Training_Time_Sec']:.4f} segundos\n")
    
    # Desplegar los gráficos comparativos
    plot_comparisons(df_metrics)

    # EXPORTACIÓN DE ARTEFACTOS PARA EL AVANCE 4

    # Creamos el directorio models si no existe en la raíz
    os.makedirs("models", exist_ok=True)

     # Usar nombres genéricos y estables para los artefactos
    model_path = os.path.join("models", "best_model.joblib")
    preprocessor_path = os.path.join("models", "preprocessor.joblib")

    # Guardar el modelo ganador y el preprocesador
    joblib.dump(best_model_object, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    # Mostrar las rutas estandarizadas
    print(
        f"[MLOps INFO] Modelo ganador exportado: "
        f"{model_path.replace(os.sep, '/')}"
    )
    print(
        f"[MLOps INFO] Preprocesador exportado: "
        f"{preprocessor_path.replace(os.sep, '/')}"
    )

    # ==================================================================

    #mejor modelo y el preprocesador entrenado
    return best_model_object, preprocessor

if __name__ == "__main__":
    main()