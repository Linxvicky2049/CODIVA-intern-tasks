# Decision Tree Classification

## Codveda Technologies Machine Learning Internship

**Level:** Level 2  
**Task:** Decision Tree Classification  
**Status:** Completed  
**Language:** Python

## 1. Project Overview

This project implements a Decision Tree Classification model to predict customer churn using the customer churn dataset provided for the Codveda Technologies Machine Learning Internship.

The objective is to classify customers according to whether they are likely to discontinue a service.

The project includes data preparation, model training, hyperparameter tuning, performance evaluation, feature importance analysis, and visualization.

## 2. Dataset

The project uses the provided Churn Prediction dataset.

- Training dataset: `churn-bigml-80.csv`
- Testing dataset: `churn-bigml-20.csv`
- Training records: 2,666
- Testing records: 667
- Features: 19
- Target variable: `Churn`

The dataset contains customer information, including account length, service usage, international plans, voice mail plans, and customer service calls.

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
- Used median imputation for numerical features.
- Used most-frequent imputation for categorical features.
- Applied One-Hot Encoding to categorical variables.
- Used a Scikit-learn ColumnTransformer and Pipeline.

### Model Development
- Implemented a DecisionTreeClassifier.
- Applied balanced class weights to account for class imbalance.
- Used GridSearchCV for hyperparameter optimization.
- Applied five-fold Stratified Cross-Validation.
- Selected the best-performing model using F1 score.

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

The final model achieved the following results on the provided test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 94.30% |
| Precision | 84.34% |
| Recall | 73.68% |
| F1 Score | 78.65% |
| ROC-AUC | 84.67% |
| Cross-Validation F1 | 76.88% |

### Best Hyperparameters

```python
{
    "criterion": "entropy",
    "max_depth": 7,
    "min_samples_leaf": 1,
    "min_samples_split": 5
}
```

## 6. Visualizations

The project generates the following visualizations:

1. Confusion Matrix
2. ROC Curve
3. Feature Importance Chart
4. Decision Tree Structure

These visualizations help interpret model performance and understand which features contribute to customer churn predictions.

## 7. Project Structure

```text
Task-2-Decision-Tree/
│
├── README.md
│
├── src/
│   └── decision_tree.py
│
└── outputs/
    ├── evaluation_metrics.csv
    ├── classification_report.txt
    ├── predictions.csv
    ├── feature_importance.csv
    ├── decision_tree_model.joblib
    │
    └── visualizations/
        ├── confusion_matrix.png
        ├── roc_curve.png
        ├── feature_importance.png
        └── decision_tree_structure.png
```

## 8. How to Run

Activate the project virtual environment:

```powershell
.\env\Scripts\Activate.ps1
```

Run the model:

```powershell
python "Level-2\Task-2-Decision-Tree\src\decision_tree.py"
```

## 9. Conclusion

The Decision Tree model achieved 94.30% accuracy and an F1 score of 78.65% on the provided test dataset.

Hyperparameter tuning helped identify a suitable tree configuration, while feature importance analysis provided additional insight into the factors influencing customer churn predictions.

The project demonstrates the application of supervised machine learning, classification, preprocessing pipelines, cross-validation, and model evaluation.

**Developed as part of the Codveda Technologies Machine Learning Internship.**