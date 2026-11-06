# Random Forest Classification

## Overview

This project implements a Random Forest Classification model to predict customer churn using the Churn BigML dataset.

The objective is to train and evaluate an ensemble machine-learning model capable of identifying customers who are likely to discontinue a service.

## Dataset

The project uses two CSV files:

- `churn-bigml-80.csv`: Training dataset.
- `churn-bigml-20.csv`: Testing dataset.

The dataset contains customer information, including account details, service usage, international plans, customer service calls, and churn status.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Methodology

1. Loaded the training and testing datasets.
2. Inspected missing values and duplicate records.
3. Separated numerical and categorical features.
4. Applied median imputation to numerical features.
5. Applied most-frequent imputation and one-hot encoding to categorical features.
6. Built a preprocessing and classification pipeline.
7. Trained a Random Forest classifier.
8. Used GridSearchCV with stratified cross-validation to identify suitable hyperparameters.
9. Evaluated the selected model using classification metrics.
10. Saved the trained model, predictions, reports, and visualizations.

## Best Model Parameters

The selected model was determined through hyperparameter tuning.

## Evaluation Results

| Metric | Score |
|---|---:|
| Accuracy | 94.75% |
| Precision | 92.86% |
| Recall | 68.42% |
| F1 Score | 78.79% |
| ROC-AUC | 93.13% |
| Cross-Validation F1 | 79.15% |

## Results Interpretation

The model achieved 94.75% accuracy and a ROC-AUC score of 93.13%.

Its precision of 92.86% indicates that most customers predicted to churn were actual churn cases.

The recall of 68.42% indicates that the model identified approximately 68% of the actual churn cases. This leaves room for improvement in identifying more customers at risk of leaving.

## Project Structure

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
    ├── random_forest_model.joblib
    ├── feature_importance.csv
    ├── model_summary.txt
    │
    └── visualizations/
        ├── confusion_matrix.png
        ├── roc_curve.png
        └── feature_importance.png
```

## How to Run

From the project root, activate the virtual environment:

```powershell
.\env\Scripts\Activate.ps1
```

Run the model:

```powershell
python "Level-3\Task-1-Random-Forest\src\random_forest.py"
```

## Conclusion

The Random Forest classifier successfully learned patterns associated with customer churn and achieved strong overall classification performance.

This project demonstrates data preprocessing, ensemble learning, hyperparameter optimization, model evaluation, and model persistence.