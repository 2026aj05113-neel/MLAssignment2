# Agricultural Yield Category Prediction

**Student name:** NEELKAMAL  
**Roll number:** 2026aj05113  
**Course:** M.Tech (AIML) — Machine Learning

## a. Problem statement

This project implements five classification models to predict the agricultural
`Yield Category` (`Low`, `Medium`, or `High`). It evaluates each model using
accuracy, AUC, precision, recall, F1 score, and Matthews correlation coefficient
(MCC), and demonstrates the saved models in a Streamlit application.

## b. Dataset description

The supplied CSV contains 1,239 instances, 14 input features, and one target
column, `Yield Category`. The input data contains numeric agricultural,
environmental, and production variables, plus the categorical `Area` field.
The classes are nearly balanced: Low (414), High (413), and Medium (412).

The CSV was supplied from a public Kaggle dataset. The source link provided for
this project is:
https://www.kaggle.com/datasets/umeradnaan/extinction-of-a-species-data

The URL should be checked before final submission because its title appears to
refer to extinction data, while this project CSV contains agricultural yield
data. The dataset source is credited; the preprocessing, model training,
evaluation, application, and visual presentation were completed for this
assignment.

The data was split with stratification and `random_state=42`. Fifty percent
was held out as `test_data.csv`, giving 619 training instances and 620 test
instances; the app evaluates that file without retraining.
Missing numeric values are median-imputed, missing categorical values are
mode-imputed, categorical values are one-hot encoded, and numeric values are
standardized inside each saved pipeline.

## c. GitHub Repository Link

[GitHub repository](https://github.com/2026aj05113-neel/MLAssignment2/tree/final_submission)

Branch: `final_submission`

## d. Models used

The five required classifiers are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine

### Comparison table

The following values were produced by `train_models.py` using the stratified
held-out test set (`test_data.csv`, 620 instances).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7839 | 0.9194 | 0.7859 | 0.7839 | 0.7847 | 0.6760 |
| Decision Tree | 0.7145 | 0.7859 | 0.7142 | 0.7145 | 0.7143 | 0.5718 |
| KNN | 0.7984 | 0.9345 | 0.7958 | 0.7984 | 0.7960 | 0.6986 |
| Naive Bayes | 0.6548 | 0.8797 | 0.7640 | 0.6548 | 0.5597 | 0.5431 |
| Support Vector Machine | 0.7855 | 0.9083 | 0.7907 | 0.7855 | 0.7866 | 0.6796 |

### Observations

The observations below are based on the verified `metrics.json` results. The
overall winner is selected using the test-set metrics, with particular
attention to weighted F1 and MCC rather than accuracy alone.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with 0.7839 accuracy and 0.7847 F1; better than the Decision Tree and Naive Bayes. |
| Decision Tree | Reached 0.7145 accuracy, but its AUC and MCC were below the other models. |
| KNN | Best overall result: highest accuracy (0.7984), AUC (0.9345), and MCC (0.6986). |
| Naive Bayes | Lowest accuracy (0.6548) and F1 (0.5597), although its AUC was 0.8797. |
| Support Vector Machine | Strong performance with 0.7855 accuracy, 0.7866 F1, and 0.6796 MCC. |
| Overall winner | KNN, because it achieved the strongest accuracy, AUC, F1, and MCC on the held-out test data. |

## Project files

- `train_models.py`: preprocessing, training, evaluation, and model saving.
- `app.py`: Streamlit frontend with upload, model selection, metrics, confusion matrix, and classification report.
- `requirements.txt`: deployment dependencies.
- `full_dataset.csv`: complete dataset (1,239 instances).
- `train_split.csv`: 619 training instances (50% stratified split, random_state=42).
- `test_data.csv`: 620 held-out test instances used by the app.
- `model/`: saved pipelines for all five classifiers.

## Streamlit application features

- CSV upload option for test data.
- Automatic target-column detection from the uploaded CSV.
- Model selection dropdown for all five saved models.
- Accuracy, AUC, precision, recall, F1, and MCC results in table form.
- Confusion matrix and classification report.
- No model retraining after test-data upload.

The confusion matrix shows correct predictions on the diagonal and
misclassifications outside the diagonal. The classification report shows
precision, recall, F1 score, and support for each class.

## Run locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Deployment

Deploy `app.py` from the `final_submission` branch of this GitHub repository
using Streamlit Community Cloud.

[Live Streamlit application](https://mlassignment2-b87z8zvsxwyqgqfgkmgsaw.streamlit.app/)

## Submission checklist

- [x] Complete source code included.
- [x] `requirements.txt` included.
- [x] `README.md` includes dataset details, metrics, and observations.
- [x] `test_data.csv` included.
- [x] Saved models for all five classifiers included.
- [x] Streamlit app includes CSV upload and model selection.
- [x] Streamlit app displays all required metrics.
- [x] Streamlit app displays a confusion matrix and classification report.
- [x] GitHub repository URL added.
- [x] Streamlit Community Cloud URL added.
- [ ] BITS Virtual Lab screenshot added to the final submission PDF.
- [ ] README content added to the final submission PDF.
