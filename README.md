# PRCP-1003 — Customer Transaction Prediction

## Project Overview

This project focuses on predicting whether a customer will make a transaction in the future using machine learning techniques.

The dataset contains anonymized customer information with 200 numerical features, along with a customer ID and a binary target variable.

The main objective is to build and compare different machine learning models and identify a suitable model for predicting potential future transactions.

## Dataset

The dataset contains:

- 200,000 customer records
- 200 anonymized numerical features
- `ID_code` — unique customer identifier
- `target` — binary target variable
  - `0` = Customer is not expected to transact
  - `1` = Customer is expected to transact

The dataset contains no missing values and no duplicate rows.

Due to the anonymized nature of the features, detailed feature-level interpretation is limited.

## Project Objectives

The main objectives of this project are:

1. Perform data analysis and quality checks.
2. Prepare the dataset for machine learning.
3. Build multiple classification models.
4. Compare model performance using suitable evaluation metrics.
5. Handle the class imbalance problem.
6. Tune the classification threshold.
7. Select the best model for future transaction prediction.

## Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Random Forest
- HistGradientBoosting Classifier

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

Since the target variable is imbalanced, accuracy alone was not considered sufficient for model selection.

## Confusion Matrix

![Confusion Matrix](<Confusion Matrix.png>)

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7834 | 0.2865 | 0.7754 | 0.4184 | 0.8599 | 0.5004 |
| Random Forest | 0.8995 | 0.0000 | 0.0000 | 0.0000 | 0.7791 | 0.2999 |
| HistGradientBoosting (Default) | 0.9144 | 0.8039 | 0.1958 | 0.3149 | 0.8813 | 0.5545 |
| HistGradientBoosting (Tuned) | 0.9028 | 0.5152 | 0.5488 | 0.5314 | 0.8813 | 0.5545 |

## ROC Curve

![ROC Curve](<ROC Curve.png>)

## Best Model

HistGradientBoosting was selected as the best production candidate after threshold tuning.

The default classification threshold resulted in high precision but relatively low recall. After threshold tuning, the threshold was reduced to approximately **0.2252**.

This improved the model performance for identifying potential customers:

- Precision: **0.5152**
- Recall: **0.5488**
- F1 Score: **0.5314**

The tuned model provides a better balance between precision and recall compared with the default threshold.

## Precision-Recall Curve

![Precision-Recall Curve](<Precision-Recall Curve.png>)

## Key Challenge

One of the main challenges was class imbalance.

Approximately 90% of customers belong to class 0, while only about 10% belong to class 1. Because of this imbalance, accuracy can be misleading.

For example, a model can achieve high accuracy while failing to identify positive customers effectively.

Therefore, Precision, Recall, F1 Score, ROC-AUC and PR-AUC were considered along with accuracy.

## Conclusion

This project demonstrates how machine learning can be used to predict future customer transactions.

Three classification approaches were compared. HistGradientBoosting achieved the strongest overall ranking performance, and threshold tuning significantly improved its ability to identify positive customers.

The tuned HistGradientBoosting model is therefore recommended as the production candidate based on the validation results.

For a real production deployment, the model should additionally be evaluated on an unseen test dataset and validated against business costs and benefits before deployment.

## Project Files

- `PRCP-1003-Customer-Transaction-Prediction.ipynb` — Complete analysis, model training, evaluation and conclusion.
- `README.md` — Project documentation.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Author

**Ramzan**

Machine Learning / Data Science Project
