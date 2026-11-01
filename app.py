"""Streamlit interface for evaluating the saved classification models."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET = "Yield Category"
STUDENT_NAME = "NEELKAMAL"
ROLL_NUMBER = "2026aj05113"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_models() -> dict[str, object]:
    return {
        name: joblib.load(ROOT / "model" / filename)
        for name, filename in MODEL_FILES.items()
    }


def required_features(model: object) -> list[str]:
    """Return the feature names recorded when the model was trained."""
    features = getattr(model, "feature_names_in_", None)
    if features is None:
        raise ValueError("The saved model does not contain its training feature names.")
    return list(features)


def main() -> None:
    st.set_page_config(page_title="Machine Learning Assignment", layout="wide")
    st.title("Machine Learning Assignment")
    st.subheader("Classification model evaluation")
    st.markdown(f"**Student name:** {STUDENT_NAME}  \n**Roll number:** {ROLL_NUMBER}")
    st.write(
        "Upload the held-out test CSV, select one of five saved classifiers, "
        "and review its evaluation results. The app does not retrain models."
    )

    with st.container(horizontal=True, vertical_alignment="bottom"):
        uploaded = st.file_uploader("Upload test data only (CSV)", type="csv")
        model_name = st.selectbox("Select a model", list(MODEL_FILES))

    st.caption(
        "Assignment requirement: upload only test data. The test CSV must contain "
        "all required input features and one target column. The app detects the "
        "target column from the uploaded data; the 14 input columns must match the "
        "saved models. The app loads the saved model and does not retrain on uploaded data."
    )

    if uploaded is None:
        st.info("Upload test_data.csv to begin.")
        return

    data = pd.read_csv(uploaded)
    data.columns = data.columns.str.strip()
    if "Area" in data.columns:
        data["Area"] = data["Area"].astype("string").str.strip()

    models = load_models()
    model = models[model_name]
    expected_features = required_features(model)
    missing_features = [column for column in expected_features if column not in data.columns]
    if missing_features:
        st.error(
            "This CSV does not match the saved models. Missing required columns: "
            + ", ".join(missing_features)
        )
        st.info(
            "Upload test_data.csv generated from the same dataset used to train the models."
        )
        return

    extra_columns = [column for column in data.columns if column not in expected_features]
    if DEFAULT_TARGET in extra_columns:
        target_column = DEFAULT_TARGET
    elif len(extra_columns) == 1:
        target_column = extra_columns[0]
    elif not extra_columns:
        st.error(
            "The uploaded test CSV must contain a target column so the app can "
            "calculate evaluation metrics."
        )
        return
    else:
        target_column = st.selectbox("Select the target column", extra_columns)

    st.caption(f"Target column detected from uploaded data: `{target_column}`")

    # Preserve the exact feature order used when training the saved model.
    features = data[expected_features]
    actual = data[target_column]

    try:
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
    except Exception as exc:
        st.error(f"The uploaded columns do not match the saved model: {exc}")
        return

    labels = sorted(actual.unique().tolist())
    model_classes = list(model.classes_)
    metrics = {
        "Accuracy": accuracy_score(actual, predictions),
        "AUC": roc_auc_score(
            actual,
            probabilities,
            labels=model_classes,
            multi_class="ovr",
            average="weighted",
        ),
        "Precision": precision_score(actual, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(actual, predictions, average="weighted", zero_division=0),
        "F1 Score": f1_score(actual, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(actual, predictions),
    }

    st.subheader(f"{model_name} results")
    metrics_table = pd.DataFrame(
        {"Metric": list(metrics), "Value": [f"{value:.4f}" for value in metrics.values()]}
    )
    st.dataframe(metrics_table, hide_index=True, width="stretch")

    st.subheader("Confusion matrix")
    matrix = confusion_matrix(actual, predictions, labels=labels)
    st.dataframe(
        pd.DataFrame(matrix, index=labels, columns=labels), width="stretch"
    )

    st.subheader("Classification report")
    report = classification_report(actual, predictions, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")


if __name__ == "__main__":
    main()
