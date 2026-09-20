
import mlflow
import mlflow.sklearn

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATA_PATH = "data/bank-marketing-clean.csv"
MODEL_PATH = "src/propensity_model.pkl"

EXPERIMENT_NAME = "Datathon - Propensity Model"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ITER = 1000


# ============================================================
# FEATURES
# ============================================================

features = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

df = pd.read_csv(DATA_PATH, sep=";")

X = df[features]
y = df["converted"]


# ============================================================
# SEPARAÇÃO TREINO/TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
    ]
)


# ============================================================
# MODELO
# ============================================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor,
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=MAX_ITER
        ),
    ),
])


# ============================================================
# TREINAMENTO E AVALIAÇÃO
# ============================================================

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="logistic-regression-baseline"):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    # Registro dos parâmetros
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "max_iter": MAX_ITER,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "features_count": len(features),
        "removed_feature": "duration",
    })

    # Registro das métricas
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    })

    # Registro do modelo como artefato
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="propensity_model",
    )

    # Salva também no caminho utilizado pela API
    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("Experimento registrado no MLflow.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")