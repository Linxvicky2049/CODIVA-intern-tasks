# Random Forest Classification

## Codveda Technologies Machine Learning Internship

**Level:** Level 3  
**Task:** Random Forest Classification  
**Status:** Completed  
**Programming Language:** Python

## 1. Project Overview

This project implements a Random Forest Classification model to predict customer churn using the customer churn dataset provided by Codveda Technologies.

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to improve prediction performance, reduce overfitting, and produce more reliable classifications.

The objective is to identify customers who are likely to discontinue a service.

## 2. Dataset

The project uses the Codveda-provided Churn Prediction dataset.

- Training dataset: `churn-bigml-80.csv`
- Testing dataset: `churn-bigml-20.csv`
- Training records: 2,666
- Testing records: 667
- Input features: 19
- Target variable: `Churn`

The dataset includes customer account information, service usage, international plans, voice mail plans, and customer service interactions.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## 4. Methodology

### Data Preparation
- Loaded the provided training and testing datasets.
- Checked missing values and duplicate records.
- Removed duplicate records.
- Separated input features from the target variable.
- Identified numerical and categorical features.

### Data Preprocessing
- Applied median imputation to numerical features.
- Applied most-frequent imputation to categorical features.
- Used One-Hot Encoding for categorical variables.
- Implemented preprocessing using ColumnTransformer and Pipeline.

### Model Development
- Implemented RandomForestClassifier.
- Applied balanced class weights.
- Used GridSearchCV for hyperparameter tuning.
- Applied five-fold Stratified Cross-Validation.
- Selected the best model using the F1 score.

### Model Evaluation
The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Classification Report
- Confusion Matrix

## 5. Model Performance

The following metrics are generated after model evaluation:

| Metric | Score |
|---|---:|
| Accuracy | Add actual result |
| Precision | Add actual result |
| Recall | Add actual result |
| F1 Score | Add actual result |
| ROC-AUC | Add actual result |
| Cross-Validation F1 | Add actual result |

### Best Hyperparameters

The optimal Random Forest configuration is automatically selected through GridSearchCV and saved in:

`outputs/classification_report.txt`

## 6. Feature Importance

The project calculates feature importance to identify which customer attributes contribute most to the model's predictions.

The top 15 features are saved in:

`outputs/feature_importance.csv`

## 7. Visualizations

The project generates:

1. Confusion Matrix
2. ROC Curve
3. Feature Importance Chart

## 8. Project Structure

```text
Task-1-Random-Forest/
│
├── README.md
│
├── src/
│   └── random_forest.py
│
└── outputs/
    ├── evaluation_metrics.csv
    ├── classification_report.txt
    ├── predictions.csv
    ├── feature_importance.csv
    ├── random_forest_model.joblib
    ├── model_summary.txt
    │
    └── visualizations/
        ├── confusion_matrix.png
        ├── roc_curve.png
        └── feature_importance.png
```

## 9. How to Run

Activate the project environment:

```powershell
.\env\Scripts\Activate.ps1
```

Run the model:

```powershell
python "Level-3\Task-1-Random-Forest\src\random_forest.py"
```

## 10. Conclusion

This project demonstrates the application of ensemble learning using Random Forest for customer churn classification.

It covers data preprocessing, model development, hyperparameter optimization, cross-validation, model evaluation, and feature importance analysis.

Developed as part of the Codveda Technologies Machine Learning Internship.