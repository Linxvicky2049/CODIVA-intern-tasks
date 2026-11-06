from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "1) iris.csv"

OUTPUT_DIR = TASK_DIR / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

print("\n========== KNN CLASSIFICATION ==========\n")

print("Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)

print("\nDataset preview:")
print(df.head())

print("\nDataset information:")
print(df.info())


# ==========================================================
# 3. DATA CLEANING
# ==========================================================

print("\nCleaning dataset...")

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:", df.duplicated().sum())

df = df.drop_duplicates()
df = df.dropna()

print("\nCleaned dataset shape:", df.shape)


# ==========================================================
# 4. DEFINE FEATURES AND TARGET
# ==========================================================

TARGET_COLUMN = "species"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' is missing."
    )

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
    raise ValueError("All Iris input features must be numeric.")

print("\nFeatures:")
print(list(X.columns))

print("\nTarget classes:")
print(y.value_counts())


# ==========================================================
# 5. SPLIT DATASET
# ==========================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================================
# 6. BUILD KNN PIPELINE
# ==========================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])


# ==========================================================
# 7. HYPERPARAMETER TUNING
# ==========================================================

print("\nSearching for the best KNN parameters...")

parameter_grid = {
    "knn__n_neighbors": list(range(1, 21)),
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    cv=cross_validation,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print(
    "\nBest cross-validation accuracy:",
    round(grid_search.best_score_, 4)
)


# ==========================================================
# 8. MODEL EVALUATION
# ==========================================================

print("\nEvaluating model...")

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

report = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(report)


# ==========================================================
# 9. SAVE EVALUATION METRICS
# ==========================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Best Cross Validation Accuracy"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        grid_search.best_score_
    ]
})

metrics.to_csv(
    OUTPUT_DIR / "evaluation_metrics.csv",
    index=False
)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("KNN CLASSIFICATION REPORT\n\n")
    file.write(f"Best Parameters: {grid_search.best_params_}\n\n")
    file.write(report)


# ==========================================================
# 10. SAVE PREDICTIONS
# ==========================================================

predictions = pd.DataFrame({
    "Actual Species": y_test.values,
    "Predicted Species": y_pred,
    "Correct Prediction": y_test.values == y_pred
})

predictions.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)


# ==========================================================
# 11. CONFUSION MATRIX
# ==========================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=best_model.classes_
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)

display.plot(cmap="Blues", values_format="d")

plt.title("KNN Classification Confusion Matrix")
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 12. ACCURACY VS K VISUALIZATION
# ==========================================================

print("Generating accuracy comparison chart...")

cv_results = pd.DataFrame(grid_search.cv_results_)

accuracy_by_k = (
    cv_results
    .groupby("param_knn__n_neighbors")["mean_test_score"]
    .max()
)

plt.figure(figsize=(10, 6))

plt.plot(
    accuracy_by_k.index.astype(int),
    accuracy_by_k.values,
    marker="o"
)

plt.title("KNN Cross-Validation Accuracy vs K")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Cross-Validation Accuracy")
plt.xticks(range(1, 21))
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "accuracy_vs_k.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 13. SAVE TRAINED MODEL
# ==========================================================

joblib.dump(
    best_model,
    OUTPUT_DIR / "knn_model.joblib"
)


# ==========================================================
# 14. FINAL VALIDATION
# ==========================================================

expected_files = [
    OUTPUT_DIR / "evaluation_metrics.csv",
    OUTPUT_DIR / "classification_report.txt",
    OUTPUT_DIR / "predictions.csv",
    OUTPUT_DIR / "knn_model.joblib",
    VISUALIZATION_DIR / "confusion_matrix.png",
    VISUALIZATION_DIR / "accuracy_vs_k.png"
]

print("\n========== OUTPUT VALIDATION ==========")

all_outputs_exist = True

for output_file in expected_files:
    exists = output_file.exists()

    print(
        f"{output_file.name}: "
        f"{'CREATED' if exists else 'MISSING'}"
    )

    if not exists:
        all_outputs_exist = False

print("\n========== FINAL RESULT ==========")

print("All output files created:", all_outputs_exist)
print("KNN task completed successfully!" if all_outputs_exist else
      "Some output files are missing. Check the errors above.")

print("\nOutput directory:")
print(OUTPUT_DIR)