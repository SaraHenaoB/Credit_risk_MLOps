# Credit_risk_MLOps
Sistema de monitoreo de datos e inferencia de riesgo crediticio mediante Machine Learning.

[![Live Demo](https://img.shields.io/badge/🚀_Demo_en_vivo-Streamlit-FF4B4B?style=for-the-badge)](https://creditriskmlops-wxj6b2kakwswk6hebewmkc.streamlit.app/)


Proyecto de Machine Learning y MLOps orientado a la predicción del comportamiento de pago de créditos, con un pipeline de entrenamiento, monitoreo de data drift y despliegue de un modelo de inferencia mediante FastAPI y Docker.

Nota: El dashboard de monitoreo y la API de inferencia son componentes separados. La disponibilidad pública de la aplicación de Streamlit no implica que la API FastAPI esté publicada en internet.


## Contexto del negocio

La correcta gestión del riesgo crediticio exige herramientas capaces de predecir la probabilidad de mora de nuevos usuarios. El núcleo de este proyecto consiste en desarrollar un modelo predictivo mediante técnicas de aprendizaje automático, utilizando información histórica de créditos para anticipar con precisión el comportamiento financiero de los clientes. 

Aunque los algoritmos de Machine Learning resuelven la predicción inicial, el verdadero reto en producción radica en controlar la degradación del modelo debido a cambios en el entorno (*data drift*). Para afrontarlo, este proyecto implementa un ecosistema integral de **MLOps** estructurado en módulos independientes que centraliza la transformación de datos, el entrenamiento automatizado, el monitoreo estadístico y el despliegue del modelo en un contenedor Docker orquestado con FastAPI.

Debido a que la empresa opera bajo un esquema estructurado de proyectos, esta iniciativa se ciñe estrictamente a una arquitectura de carpetas predefinida e inalterable. Esta rigidez es fundamental, ya que los procesos de despliegue a producción están completamente automatizados a través de pipelines de validación en **Jenkins**; cualquier alteración en la organización de los archivos generaría fallos en el flujo de integración y retrasos significativos en el paso a producción. De este modo, la infraestructura no solo valida el rendimiento técnico actual, sino que asegura un ciclo de vida reproducible, auditable y alineado con los estándares operativos de la organización.


---

## Objetivo del Negocio

Predecir la probabilidad de que un cliente pague un crédito financiero a tiempo o entre en mora:
*   **Variable Objetivo ('Pago_atiempo')**: 
    *   1: El cliente paga a tiempo.
    *   0: El cliente entra en mora (Clase Crítica).
*   **Desbalanceo Crítico**: 95% pertenecen a la clase 1 y solo 5% a la clase 0. El pipeline tiene en cuenta este desbalanceo para el entrenamiento de modelos.

---

## Arquitectura del Software y Componentes

El proyecto está modularizado en scripts de Python independientes para asegurar mantenibilidad y escalabilidad en entornos productivos:
```
Credit_risk_MLOps/
├── src/
│   ├── cargar_datos.py 
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_monitoring.py
│   ├── model_deploy.py
│   ├── metricas_comparativas_modelos.png
│   └── models/
│       ├── best_model.joblib
│       └── preprocessor.joblib
├── Base_de_datos.xlsx
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Componentes principales
```
Archivo                         |    Responsabilidad
-------------------------------------------------------------------------------------------------------------------------
cargar_datos.py                 |    Carga el conjunto de datos utilizado por el pipeline.
ft_engineering.py               |    Prepara los datos, controla la fuga de información y construye el preprocesamiento.
model_training_evaluation.py    |    Entrena y compara modelos, selecciona el mejor candidato y guarda los artefactos.
model_monitoring.py             |    Implementa el dashboard Streamlit para el monitoreo de data drift.
model_deploy.py                 |    Expone el modelo entrenado mediante una API FastAPI para realizar predicciones.
src/models/                     |    Contiene el modelo seleccionado y el preprocesador serializados.
Dockerfile                      |    Define la imagen que contiene el entorno y el código para ejecutar la API.
requirements.txt                |    Declara las dependencias de Python.
```

## Ingeniería de características y preprocesamiento
El módulo ft_engineering.py prepara los datos para el entrenamiento y la inferencia, procurando mantener un tratamiento consistente entre ambos procesos.
### Prevención de fuga de datos
Se excluyen variables que podrían incorporar información posterior al otorgamiento del crédito o introducir fuga de información, entre ellas:
- saldo_mora
- saldo_mora_codeudor
- saldo_total
- saldo_principal
- puntaje

La variable objetivo Pago_atiempo tampoco debe formar parte de las variables de entrada del modelo.
### Validación temporal
Los registros se ordenan cronológicamente mediante fecha_prestamo y se separan en conjuntos de entrenamiento y prueba, con una proporción de 80 % y 20 %, respectivamente, utilizando shuffle=False.

Este enfoque permite evaluar el modelo con registros posteriores a los utilizados para entrenarlo y reduce el riesgo de fuga temporal.
### Pipeline de preprocesamiento
El preprocesamiento se integra mediante ColumnTransformer:
- Variables numéricas: imputación por mediana, indicadores de valores faltantes cuando corresponde y escalamiento mediante StandardScaler.
- Variables nominales: imputación por la categoría más frecuente y codificación con OneHotEncoder.
- Variables ordinales: tratamiento explícito de valores faltantes, codificación mediante OrdinalEncoder y escalamiento cuando corresponde.
El preprocesador ajustado durante el entrenamiento se guarda como preprocessor.joblib para reutilizar las mismas transformaciones durante la inferencia.

## Entrenamiento y evaluación de modelos
El módulo model_training_evaluation.py entrena y compara distintos algoritmos de clasificación para seleccionar el modelo candidato con mejor rendimiento según la métrica definida en el pipeline.
### Modelos evaluados
- Regresión logística (Logistic Regression).
- Bosque aleatorio (Random Forest).
- XGBoost.
- LightGBM.

### Evaluación
Debido al desbalanceo de clases, se consideran métricas como F1-Macro y ROC-AUC, además de la comparación del rendimiento entre entrenamiento y prueba.

El pipeline selecciona el modelo automáticamente a partir de los resultados obtenidos, en lugar de depender de un algoritmo fijado manualmente.

Los artefactos utilizados en el despliegue son:
- best_model.joblib: modelo seleccionado.
- preprocessor.joblib: transformaciones ajustadas durante el entrenamiento.
- metricas_comparativas_modelos.png: visualización comparativa de las métricas calculadas.
La selección del modelo y sus métricas deben corresponder a la última ejecución documentada del pipeline.

## Monitoreo de datos (Data Drift)
El módulo model_monitoring.py contiene un dashboard desarrollado con Streamlit para comparar los datos de referencia con los datos recientes y detectar posibles cambios en sus distribuciones.
### Metodología
El proceso ordena los registros por fecha y establece una población de referencia histórica (80 %) y una población reciente (20 %). Para las pruebas de monitoreo se utiliza una simulación de cambios en determinadas variables financieras.

Entre las modificaciones simuladas se encuentran una reducción del 30 % en salario_cliente y un incremento del 40 % en capital_prestado. Estos cambios permiten evaluar si el sistema identifica desviaciones conocidas; no representan necesariamente cambios observados en producción.
### Pruebas estadísticas

- Kolmogorov-Smirnov (KS): compara distribuciones de variables numéricas.
- Chi-cuadrado: evalúa diferencias en las distribuciones de variables categóricas.
- Population Stability Index (PSI) y Jensen-Shannon divergence (JS): complementan el análisis de cambios distributivos cuando están calculados por el módulo.

Un p-valor inferior a 0,05 se utiliza como criterio de detección para las pruebas estadísticas correspondientes. Los indicadores PSI y JS deben interpretarse de acuerdo con los umbrales implementados en el código.

### Visualización y alertas
El dashboard permite consultar los resultados del monitoreo mediante tablas y gráficos. Las variables identificadas con desviaciones se resaltan visualmente para facilitar su revisión.

La lógica de alerta agregada contempla una señal de revisión o posible reentrenamiento cuando más de dos variables presentan drift según las reglas configuradas.

Aplicación publicada: Se encuentra disponible en el link al inicio del README

El monitoreo de data drift no demuestra por sí solo que el rendimiento predictivo haya disminuido. Para evaluar esa degradación también se necesitarían etiquetas reales posteriores y métricas de desempeño actualizadas.

## API de inferencia con FastAPI
El archivo model_deploy.py implementa el servicio de inferencia. Carga el modelo y el preprocesador serializados, valida los datos de entrada, aplica las transformaciones y devuelve las predicciones en formato JSON.
### Endpoints
Método	Endpoint	Descripción

```
GET	/   	        Confirma que la API está activa.
GET	/docs	        Abre la documentación interactiva de Swagger.
POST	/predict	Recibe registros en JSON y devuelve las predicciones.
```

El endpoint /predict admite predicciones por lotes, de modo que varios registros pueden enviarse en una única solicitud.

La respuesta incluye la clase predicha, su interpretación y las probabilidades por clase disponibles.

La interpretación de las clases es:
- 1: paga a tiempo.
- 0: incumplimiento.

Ejemplo de solicitud
El siguiente ejemplo ilustra la estructura de entrada para un registro:

```
{
  "records": [
    {
      "tipo_credito": null,
      "capital_prestado": 5000000,
      "plazo_meses": 24,
      "edad_cliente": 30,
      "tipo_laboral": null,
      "salario_cliente": 2500000,
      "total_otros_prestamos": 0,
      "cuota_pactada": 300000,
      "puntaje_datacredito": 700,
      "cant_creditosvigentes": 1,
      "huella_consulta": null,
      "creditos_sectorFinanciero": 1,
      "creditos_sectorCooperativo": 0,
      "creditos_sectorReal": 0,
      "promedio_ingresos_datacredito": 2500000,
      "tendencia_ingresos": "Estable"
    }
  ]
}
```

Los valores son ilustrativos. Los nombres, tipos de datos y categorías enviados deben coincidir con los esperados por la API.

## Instalación y ejecución local
Requisitos previos
- Git.
- Python compatible con las dependencias declaradas.
- Docker Desktop, si se desea ejecutar el servicio en un contenedor.

### 1. Clonar el repositorio
```
git clone https://github.com/SaraHenaoB/Credit_risk_MLOps.git
cd Credit_risk_MLOps
```
### 2. Crear un entorno virtual

En Windows PowerShell:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
En macOS o Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```
python -m pip install -r requirements.txt
```
### 4. Ejecutar el dashboard de monitoreo
Desde la raíz del repositorio:
```
streamlit run src/model_monitoring.py
```
Streamlit mostrará en la terminal la dirección local para abrir el dashboard en el navegador.

### 5. Ejecutar la API de inferencia
En otra terminal, desde la raíz del repositorio y con las dependencias instaladas:
```
uvicorn src.model_deploy:app --host 127.0.0.1 --port 8000
```

Abrir la documentación interactiva en:

http://localhost:8000/docs

Desde Swagger se puede ejecutar POST /predict y comprobar la respuesta de la API con el registro de ejemplo.

## Construcción y ejecución con Docker

El proyecto incluye un Dockerfile que instala las dependencias y copia el código y los artefactos necesarios para ejecutar la API.
### 1. Construir la imagen
Desde la raíz del repositorio:
```
docker build -t credit-risk-mlops .
```

### 2. Ejecutar el contenedor
```
docker run --name credit-risk-api -p 8000:8000 credit-risk-mlops
```

La API estará disponible localmente en:
- API: http://localhost:8000/
- Documentación interactiva: http://localhost:8000/docs

### 3. Probar las predicciones

En Swagger:
1. Abrir /docs.
2. Seleccionar POST /predict.
3. Pulsar Try it out.
4. Introducir un JSON que cumpla el esquema de entrada.
5. Pulsar Execute y revisar el código HTTP y el resultado.

Para detener el contenedor, utiliza Ctrl+C en la terminal donde se está ejecutando. Si se ejecuta en segundo plano, puede detenerse mediante:
```
docker stop credit-risk-api
```
Si se necesita crear nuevamente un contenedor con el mismo nombre después de eliminarlo:
```
docker rm credit-risk-api
```
La imagen Docker contiene el servicio de inferencia. La aplicación Streamlit publicada es un componente separado; publicar el dashboard no implica que la API FastAPI también esté disponible públicamente.

## Tecnologías utilizadas
- Lenguaje: Python.
- Manipulación de datos: Pandas y NumPy.
- Machine Learning: scikit-learn, XGBoost y LightGBM.
- Monitoreo y visualización: Streamlit, SciPy y bibliotecas de visualización.
- API: FastAPI y Uvicorn.
- Serialización de modelos: Joblib.
- Contenerización: Docker.
- Control de versiones: Git y GitHub.

## Limitaciones y próximos pasos
- Validar el modelo con datos adicionales y posteriores al período de entrenamiento.
- Evaluar periódicamente las métricas predictivas por clase cuando estén disponibles las etiquetas reales.
- Definir procedimientos controlados para revisar alertas y decidir cuándo corresponde reentrenar.
- Incorporar registros, métricas operativas y seguimiento de las versiones del modelo.
- Configurar y asegurar un despliegue público de la API si se requiere consumir predicciones desde sistemas externos.
La detección de data drift es una señal de cambio en los datos y no equivale automáticamente a una caída del rendimiento del modelo. Del mismo modo, las predicciones no deben considerarse una decisión financiera definitiva sin validaciones adicionales y controles apropiados.

## 👩‍💻 Autora

Sara Henao


Data Science | Machine Learning | MLOps
