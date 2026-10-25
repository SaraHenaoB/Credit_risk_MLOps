# Importar librerías
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

from cargar_datos import cargarDatos

import warnings
warnings.filterwarnings('ignore')


def clean_data(df):
    """
    Función para para limpieza de datos corruptos
    """
    #Copia para no afectar el dataset original
    df_clean = df.copy()

    valid_values = ["Estable", "Creciente", "Decreciente"]
    df_clean["tendencia_ingresos"] = df_clean["tendencia_ingresos"].apply(
        lambda x: x if x in valid_values else None
    )
    return df_clean
  

def split_data(df):
    """
    Separación de datos de entrenamiento se ordena por fecha, de lo más antiguo a lo más reciente
    """    
    target = "Pago_atiempo"
    df_sorted = df.copy()

    if "fecha_prestamo" in df_sorted.columns:
            # Convertimos especificando el formato correcto (Mes/Día/Año Hora:Minuto)
            # errors='coerce' transforma cualquier texto corrupto en NaT para que no rompa el script
            df_sorted["fecha_prestamo"] = pd.to_datetime(
                df_sorted["fecha_prestamo"], 
                format="%m/%d/%Y %H:%M", 
                errors="coerce"
            )

    # Eliminamos filas que se hayan quedado sin fecha por corrupción (si las hay)
    df_sorted = df_sorted.dropna(subset=["fecha_prestamo"])
        
    #Ordenar cronológicamente de la más antigua a la más reciente
    df_sorted = df_sorted.sort_values(by="fecha_prestamo", ascending=True).reset_index(drop=True)

    #Eliminamos las columnas objetivo y de fecha_prestamo (La info de fecha para este caso no aporta mucho al análisis ya que 
    # se desconoce el comportamientiento del dataset y solo serían especulaciones sin rigor)para evitar ruido en el entrenamiento
    X = df_sorted.drop(
         columns=[target, "fecha_prestamo","saldo_mora", "saldo_mora_codeudor", "saldo_total", "saldo_principal", "puntaje"],
           errors="ignore")         # features
    y = df_sorted[target]           # target

    #Tener en cuenta la tenporalidad. Shuffle = False
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


#Pipelines
def preprocesing_pipeline(numerical_features, nominal_features, ordinal_features):
    """
    Construye y retorna la estructura del ColumnTransformer aplicando
    OneHotEncoder para nominales y OrdinalEncoder para ordinales
    """
    #pipeline variables numéricas
    #add_indicator=True crea la columna espejo para el 27% de nulos de forma automática 
    # y le indicará a los modelos de regresión que es un dato imputado para que no lo tome en cuenta y no se rompa el código
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])

    #pipeline variables categóricas
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("One_hot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))# Para que devuelva matriz densa tradicional de NumPy
    ])
    #Pipeline especial para la variable categorica con 27% de nulos
    order_trend = ["sin información", "Decreciente", "Estable", "Creciente"]
    ordinal_cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="sin información")),
        ("ordinal", OrdinalEncoder(categories=[order_trend]))
    ])


    #Integración
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, nominal_features),
        ("cat_ord", ordinal_cat_pipeline,ordinal_features)
    ])
    return preprocessor



def ft_engineering_pipeline(df):
    """
    Función orquestadora que ejecuta todo el proceso de ingeniería de características
    Distingue variables numéricas, nominales y ordinales
    """
    #limpieza con la función clean_data
    df = clean_data(df)

    #Separamos features y target
    target = "Pago_atiempo"
    #PURGA DE VARIABLES: Eliminamos las columnas que causan la trampa del 1.0
    columnas_fuga = [
        target, 
        "fecha_prestamo", 
        "saldo_mora", 
        "saldo_mora_codeudor", 
        "saldo_total", 
        "saldo_principal",
        "puntaje" # se elimina la variable de mayor correlación ya que no está permitiendo predecir, 
                    #probablemente sea calculada después de haber hecho el pago
    ]
    
    X = df.drop(columns=[col for col in columnas_fuga if col in df.columns], errors="ignore")        #features
    y = df[target]                                                                                   # target

    # 'promedio_ingresos_datacredito' se incluirá automáticamente aquí por ser float64
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    nominal_features = ["tipo_laboral"]
    ordinal_features = ["tendencia_ingresos"]


    print(f"Variables numéricas: {numerical_features}")
    print(f"Variables categóricas: {nominal_features}")
    print(f"Variables categóricas ordinales: {ordinal_features}")

    #Separacion de data
    X_train, X_test, y_train, y_test = split_data(df)

    #Llamar a Pipeline preprocesamiento
    preprocessor = preprocesing_pipeline(numerical_features, nominal_features, ordinal_features)

    #preprocesador aprende fit sólo de entrenamiento para evitar data leakage
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("--- Ingeniería de Características Exitosa ---")
    print(f"Dimensiones Finales - X_train: {X_train_processed.shape} | X_test: {X_test_processed.shape}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def main():
    print("Iniciando Pipeline de Ingeniería de Características desde MAIN")
    df=cargarDatos()
    print(df)

    #Ejecutar el pipeline
    X_train, X_test, y_train, y_test, preprocessor = ft_engineering_pipeline (df)

    return X_train, X_test, y_train, y_test, preprocessor

#Para ejecución directa en terminal
if __name__ == "__main__":
    main()