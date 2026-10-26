# Credit_risk_MLOps
Data monitoring and credit risk inference system using Machine Learning.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://creditriskmlops-wxj6b2kakwswk6hebewmkc.streamlit.app/)


A Machine Learning and MLOps project focused on predicting credit repayment behavior, featuring a training pipeline, data drift monitoring, and deployment of an inference model using FastAPI and Docker.

Note: The monitoring dashboard and the inference API are separate components. The public availability of the Streamlit application does not mean that the FastAPI API is published on the internet.


## Business Context

Proper credit risk management requires tools capable of predicting the probability that new users will default. The core of this project is to develop a predictive model using machine learning techniques and historical credit data to accurately anticipate customers' financial behavior. 

Although Machine Learning algorithms handle the initial prediction, the real challenge in production is controlling model degradation caused by changes in the environment (*data drift*). To address this, this project implements a comprehensive **MLOps** ecosystem structured into independent modules that centralizes data transformation, automated training, statistical monitoring, and model deployment in a Docker container orchestrated with FastAPI.

Because the company operates under a structured project framework, this initiative strictly adheres to a predefined, unalterable folder architecture. This rigidity is essential because production deployment processes are fully automated through **Jenkins** validation pipelines; any change to the file organization would cause integration workflow failures and significant delays in moving to production. In this way, the infrastructure not only validates current technical performance but also ensures a reproducible, auditable lifecycle aligned with the organization's operational standards.


---

## Business Objective

Predict the probability that a customer will repay a financial loan on time or default:
*   **Target Variable ('Pago_atiempo')**: 
    *   1: The customer pays on time.
    *   0: The customer defaults (Critical Class).
*   **Critical Class Imbalance**: 95% belong to class 1 and only 5% to class 0. The pipeline accounts for this imbalance when training models.

---

## Software Architecture and Components

The project is modularized into independent Python scripts to ensure maintainability and scalability in production environments:
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

## Main Components
```
File                            |    Responsibility
-------------------------------------------------------------------------------------------------------------------------
cargar_datos.py                 |    Loads the dataset used by the pipeline.
ft_engineering.py               |    Prepares the data, controls data leakage, and builds the preprocessing pipeline.
model_training_evaluation.py    |    Trains and compares models, selects the best candidate, and saves the artifacts.
model_monitoring.py             |    Implements the Streamlit dashboard for data drift monitoring.
model_deploy.py                 |    Exposes the trained model through a FastAPI API to make predictions.
src/models/                     |    Contains the serialized selected model and preprocessor.
Dockerfile                      |    Defines the image containing the environment and code required to run the API.
requirements.txt                |    Declares the Python dependencies.
```

## Feature Engineering and Preprocessing
The ft_engineering.py module prepares data for training and inference, aiming to maintain consistent processing across both processes.
### Data Leakage Prevention
Variables that could contain information from after the credit was granted or introduce data leakage are excluded, including:
- saldo_mora
- saldo_mora_codeudor
- saldo_total
- saldo_principal
- puntaje

The target variable Pago_atiempo must not be included among the model's input features either.
### Temporal Validation
Records are sorted chronologically by fecha_prestamo and split into training and test sets at a ratio of 80% and 20%, respectively, using shuffle=False.

This approach makes it possible to evaluate the model on records later than those used for training and reduces the risk of temporal leakage.
### Preprocessing Pipeline
Preprocessing is integrated using ColumnTransformer:
- Numeric variables: median imputation, missing-value indicators where applicable, and scaling using StandardScaler.
- Nominal variables: imputation using the most frequent category and encoding with OneHotEncoder.
- Ordinal variables: explicit handling of missing values, encoding using OrdinalEncoder, and scaling where applicable.
The preprocessor fitted during training is saved as preprocessor.joblib so the same transformations can be reused during inference.

## Model Training and Evaluation
The model_training_evaluation.py module trains and compares different classification algorithms to select the best-performing candidate model according to the metric defined in the pipeline.
Models evaluated
- Logistic Regression.
- Random Forest.
- XGBoost.
- LightGBM.
Evaluation
Due to class imbalance, metrics such as F1-Macro and ROC-AUC are considered, along with a comparison of training and test performance.

The pipeline automatically selects the model based on the results obtained, rather than relying on a manually fixed algorithm.

The artifacts used for deployment are:
- best_model.joblib: selected model.
- preprocessor.joblib: transformations fitted during training.
- metricas_comparativas_modelos.png: comparative visualization of the calculated metrics.
The model selection and its metrics should correspond to the latest documented pipeline run.

## Data Monitoring (Data Drift)
The model_monitoring.py module contains a dashboard developed with Streamlit to compare reference data with recent data and detect potential changes in their distributions.
### Methodology
The process sorts records by date and establishes a historical reference population (80%) and a recent population (20%). For monitoring tests, changes in certain financial variables are simulated.

The simulated modifications include a 30% reduction in salario_cliente and a 40% increase in capital_prestado. These changes make it possible to assess whether the system identifies known deviations; they do not necessarily represent changes observed in production.
### Statistical Tests

- Kolmogorov-Smirnov (KS): compares distributions of numeric variables.
- Chi-square: evaluates differences in the distributions of categorical variables.
- Population Stability Index (PSI) and Jensen-Shannon divergence (JS): complement the analysis of distributional changes when calculated by the module.

A p-value below 0.05 is used as the detection criterion for the corresponding statistical tests. PSI and JS indicators should be interpreted according to the thresholds implemented in the code.

### Visualization and Alerts
The dashboard allows users to review monitoring results through tables and charts. Variables identified as having deviations are highlighted visually to make them easier to review.

The aggregated alert logic issues a review or potential retraining signal when more than two variables show drift according to the configured rules.

Published application: Available at the link at the beginning of the README

Data drift monitoring alone does not demonstrate that predictive performance has decreased. Evaluating such degradation would also require subsequent ground-truth labels and updated performance metrics.

## Inference API with FastAPI
The model_deploy.py file implements the inference service. It loads the serialized model and preprocessor, validates the input data, applies the transformations, and returns predictions in JSON format.
### Endpoints
Method	Endpoint	Description

```
GET	/   	        Confirms that the API is active.
GET	/docs	        Opens the interactive Swagger documentation.
POST	/predict	Receives records in JSON and returns predictions.
```

The /predict endpoint supports batch predictions, allowing multiple records to be sent in a single request.

The response includes the predicted class, its interpretation, and the available class probabilities.

The classes are interpreted as follows:
- 1: pays on time.
- 0: default.

Example request
The following example illustrates the input structure for one record:

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

The values are illustrative. The names, data types, and categories submitted must match those expected by the API.

## Local Installation and Execution
Prerequisites
- Git.
- Python compatible with the declared dependencies.
- Docker Desktop, if you want to run the service in a container.

### 1. Clone the repository
```
git clone https://github.com/SaraHenaoB/Credit_risk_MLOps.git
cd Credit_risk_MLOps
```
### 2. Create a virtual environment

On Windows PowerShell:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
On macOS or Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```
python -m pip install -r requirements.txt
```
### 4. Run the monitoring dashboard
From the repository root:
```
streamlit run src/model_monitoring.py
```
Streamlit will display the local address in the terminal for opening the dashboard in a browser.

### 5. Run the inference API
In another terminal, from the repository root and with the dependencies installed:
```
uvicorn src.model_deploy:app --host 127.0.0.1 --port 8000
```

Open the interactive documentation at:

http://localhost:8000/docs

From Swagger, you can run POST /predict and check the API response using the example record.

## Building and Running with Docker

The project includes a Dockerfile that installs the dependencies and copies the code and artifacts required to run the API.
### 1. Build the image
From the repository root:
```
docker build -t credit-risk-mlops .
```

### 2. Run the container
```
docker run --name credit-risk-api -p 8000:8000 credit-risk-mlops
```

The API will be available locally at:
- API: http://localhost:8000/
- Interactive documentation: http://localhost:8000/docs

### 3. Test predictions

In Swagger:
1. Open /docs.
2. Select POST /predict.
3. Click Try it out.
4. Enter JSON that conforms to the input schema.
5. Click Execute and review the HTTP status code and result.

To stop the container, use Ctrl+C in the terminal where it is running. If it is running in the background, it can be stopped using:
```
docker stop credit-risk-api
```
If you need to create a container with the same name again after removing it:
```
docker rm credit-risk-api
```
The Docker image contains the inference service. The published Streamlit application is a separate component; publishing the dashboard does not mean that the FastAPI API is also publicly available.

## Technologies Used
- Language: Python.
- Data manipulation: Pandas and NumPy.
- Machine Learning: scikit-learn, XGBoost, and LightGBM.
- Monitoring and visualization: Streamlit, SciPy, and visualization libraries.
- API: FastAPI and Uvicorn.
- Model serialization: Joblib.
- Containerization: Docker.
- Version control: Git and GitHub.

## Limitations and Next Steps
- Validate the model with additional data collected after the training period.
- Periodically evaluate predictive metrics by class when ground-truth labels become available.
- Define controlled procedures for reviewing alerts and deciding when retraining is appropriate.
- Incorporate logs, operational metrics, and model version tracking.
- Configure and secure a public API deployment if predictions need to be consumed by external systems.
Data drift detection is a signal that the data has changed and does not automatically mean that model performance has declined. Likewise, predictions should not be considered a definitive financial decision without additional validation and appropriate controls.

## 👩‍💻 Author

Sara Henao


Data Science | Machine Learning | MLOps
