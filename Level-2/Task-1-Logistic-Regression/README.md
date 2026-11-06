# Customer Churn Prediction Using Logistic Regression

## 1. Project Overview

This project implements a Logistic Regression machine learning model to predict customer churn using the customer churn dataset provided for the Codveda Technologies Machine Learning Internship.

The objective is to identify customers who are likely to discontinue a service based on their account information, service usage, and subscription details.

## 2. Dataset

The project uses the supplied customer churn datasets.

| Dataset | Records |
|---|---:|
| Training dataset | 2,666 |
| Testing dataset | 667 |
| Total | 3,333 |

Target variable: `Churn`

- False: Customer remains with the service.
- True: Customer leaves the service.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## 4. Data Preprocessing

The following preprocessing operations were performed:

1. Loaded the supplied training and testing datasets.
2. Inspected dataset structure and data types.
3. Checked for missing values and duplicate records.
4. Removed duplicate records where present.
5. Separated input features from the target variable.
6. Identified numerical and categorical features.
7. Applied median imputation and StandardScaler to numerical features.
8. Applied most-frequent imputation and OneHotEncoder to categorical features.

A Scikit-learn Pipeline and ColumnTransformer were used to ensure preprocessing was fitted on the training data only.

## 5. Model Development

Logistic Regression was implemented using Scikit-learn.

Model configuration:

- Algorithm: Logistic Regression
- Maximum iterations: 2,000
- Class weighting: Balanced
- Random state: 42

The balanced class weighting helps the model account for the difference between the number of customers who churn and those who remain.

## 6. Model Evaluation

The model was evaluated using the supplied testing dataset.

| Metric | Result |
|---|---:|
| Accuracy | 77.96% |
| Precision | 36.87% |
| Recall | 76.84% |
| F1 Score | 49.83% |
| ROC-AUC | 81.84% |

### Interpretation

The model achieved 77.96% accuracy and an ROC-AUC score of 81.84%.

Its recall of 76.84% indicates that it identified a substantial proportion of customers who actually churned.

However, precision was 36.87%, indicating that some customers predicted to churn did not actually leave.

The difference between precision and recall highlights the challenge of predicting churn in an imbalanced dataset.

## 7. Generated Outputs

The project generates:

- `evaluation_metrics.csv`: Evaluation metrics.
- `classification_report.txt`: Detailed classification report.
- `predictions.csv`: Actual labels, predicted labels, and churn probabilities.
- `logistic_regression_model.joblib`: Saved trained model.
- `visualizations/confusion_matrix.png`: Confusion matrix.
- `visualizations/roc_curve.png`: Receiver Operating Characteristic curve.

## 8. How to Run

Activate the project's virtual environment:

```powershell
.\env\Scripts\Activate.ps1
```

Execute the model:

```powershell
python Level-2/Task-1-Logistic-Regression/src/logistic_regression.py
```

## 9. Conclusion

The Logistic Regression model successfully learned patterns in the customer churn dataset and generated predictions on the supplied test dataset.

The results demonstrate the importance of evaluating classification models using multiple metrics rather than accuracy alone, particularly when working with imbalanced datasets.

This project demonstrates data preprocessing, categorical encoding, feature scaling, machine learning pipeline development, binary classification, model evaluation, and model persistence.