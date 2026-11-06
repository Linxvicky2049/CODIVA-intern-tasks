# K-Nearest Neighbors (KNN) Classification

## Overview
This project implements a K-Nearest Neighbors (KNN) classification model using the Iris dataset provided for the Codveda Technologies Machine Learning Internship.

The objective is to classify iris flowers into three species using their sepal and petal measurements.

## Dataset
- Dataset: Iris Dataset
- Original records: 150
- Duplicate records removed: 3
- Final records: 147
- Input features: Sepal Length, Sepal Width, Petal Length, Petal Width
- Target variable: Species

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Methodology
1. Loaded and inspected the Iris dataset.
2. Checked for missing values and duplicate records.
3. Removed duplicate and incomplete records.
4. Separated input features from the target variable.
5. Split the dataset into 80% training and 20% testing data.
6. Applied StandardScaler through a Scikit-learn pipeline.
7. Used GridSearchCV to identify optimal KNN parameters.
8. Evaluated the trained model using classification metrics.
9. Generated visualizations and saved the trained model.

## Best Model Parameters
- Number of Neighbors (K): 13
- Distance Metric: Euclidean
- Weighting: Distance

## Evaluation Results

| Metric | Score |
|---|---:|
| Cross-validation Accuracy | 96.59% |
| Test Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |

The model correctly classified all 30 test samples.

## Project Outputs
- `evaluation_metrics.csv`: Model evaluation scores.
- `classification_report.txt`: Detailed classification report.
- `predictions.csv`: Actual and predicted species.
- `knn_model.joblib`: Saved trained model.
- `visualizations/confusion_matrix.png`: Confusion matrix.
- `visualizations/accuracy_vs_k.png`: Cross-validation accuracy comparison.

## How to Run

Activate the project's virtual environment:

```powershell
.\env\Scripts\Activate.ps1
```

Run the model:

```powershell
python Level-1/Task-3-KNN-Classification/src/knn.py
```

## Conclusion
The KNN model successfully classified the Iris dataset with 100% test accuracy on the selected 30-sample test set. Hyperparameter tuning identified K=13 with Euclidean distance and distance-based weighting as the best configuration.

This project demonstrates data cleaning, feature scaling, model training, hyperparameter optimization, classification evaluation, and model persistence.