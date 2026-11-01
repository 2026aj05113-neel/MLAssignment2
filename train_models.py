"""Train and evaluate the five classifiers required by the assignment."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "train_data.csv"
MODEL_DIR = ROOT / "model"
TEST_PATH = ROOT / "test_data.csv"
METRICS_PATH = ROOT / "metrics.json"
TARGET = "Yield Category"


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = features.select_dtypes(exclude=["object", "string"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    if "Area" in df.columns:
        df["Area"] = df["Area"].astype("string").str.strip()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.50, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(random_state=42), method="sigmoid", cv=5
        ),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    metrics: dict[str, dict[str, float]] = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(X_train)), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)

        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "auc": float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                )
            ),
            "precision": float(
                precision_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "f1": float(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "mcc": float(matthews_corrcoef(y_test, predictions)),
        }
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, MODEL_DIR / filename)

    test_data = X_test.copy()
    test_data[TARGET] = y_test
    test_data.to_csv(TEST_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Dataset shape: {df.shape}")
    print(f"Training instances: {len(X_train)}")
    print(f"Test instances: {len(X_test)}")
    print(pd.DataFrame(metrics).T.round(4).to_string())


if __name__ == "__main__":
    main()
