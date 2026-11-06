from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "Churn Prdiction Data"

TRAIN_PATH = DATA_DIR / "churn-bigml-80.csv"
TEST_PATH = DATA_DIR / "churn-bigml-20.csv"

OUTPUT_DIR = TASK_DIR / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LOAD DATASETS
# ==========================================================

print("\n========== LOGISTIC REGRESSION ==========\n")

print("Loading training dataset...")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training dataset not found: {TRAIN_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Testing dataset not found: {TEST_PATH}")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

print("\nTraining dataset preview:")
print(train_df.head())

print("\nDataset information:")
train_df.info()


# ==========================================================
# 3. DATA INSPECTION AND CLEANING
# ==========================================================

print("\nChecking missing values...")

print("\nTraining missing values:")
print(train_df.isnull().sum())

print("\nTesting missing values:")
print(test_df.isnull().sum())

print("\nTraining duplicates:", train_df.duplicated().sum())
print("Testing duplicates:", test_df.duplicated().sum())

train_df = train_df.drop_duplicates().copy()
test_df = test_df.drop_duplicates().copy()

TARGET_COLUMN = "Churn"

if TARGET_COLUMN not in train_df.columns:
    raise ValueError("Churn target column is missing from training data.")

if TARGET_COLUMN not in test_df.columns:
    raise ValueError("Churn target column is missing from testing data.")

if set(train_df.columns) != set(test_df.columns):
    raise ValueError("Training and testing datasets have different columns.")

# Align test columns to the training column order.
test_df = test_df[train_df.columns]

print("\nCleaned training shape:", train_df.shape)
print("Cleaned testing shape:", test_df.shape)


# ==========================================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================================

X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ==========================================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ==========================================================

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ==========================================================
# 6. BUILD PREPROCESSING PIPELINES
# ==========================================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==========================================================
# 7. BUILD LOGISTIC REGRESSION MODEL
# ==========================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ==========================================================
# 8. TRAIN MODEL
# ==========================================================

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ==========================================================
# 9. MAKE PREDICTIONS
# ==========================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

positive_label = model.named_steps["classifier"].classes_[1]

y_test_binary = (y_test == positive_label).astype(int)

# ==========================================================
# 10. MODEL EVALUATION
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, pos_label=positive_label, zero_division=0
)

recall = recall_score(
    y_test, y_pred, pos_label=positive_label, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, pos_label=positive_label, zero_division=0
)

roc_auc = roc_auc_score(y_test_binary, y_probability)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ==========================================================
# 11. SAVE EVALUATION RESULTS
# ==========================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
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
    file.write("LOGISTIC REGRESSION CLASSIFICATION REPORT\n\n")
    file.write(report)


# ==========================================================
# 12. SAVE PREDICTIONS
# ==========================================================

predictions = pd.DataFrame({
    "Actual Churn": y_test.values,
    "Predicted Churn": y_pred,
    "Churn Probability": y_probability,
    "Correct Prediction": y_test.values == y_pred
})

predictions.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)


# ==========================================================
# 13. CONFUSION MATRIX
# ==========================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.named_steps["classifier"].classes_
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.named_steps["classifier"].classes_
)

display.plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 14. ROC CURVE
# ==========================================================

print("Generating ROC curve...")

fpr, tpr, thresholds = roc_curve(
    y_test_binary,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 15. SAVE TRAINED MODEL
# ==========================================================

print("\nSaving trained model...")

joblib.dump(
    model,
    OUTPUT_DIR / "logistic_regression_model.joblib"
)


# ==========================================================
# 16. VALIDATE OUTPUT FILES
# ==========================================================

expected_files = [
    OUTPUT_DIR / "evaluation_metrics.csv",
    OUTPUT_DIR / "classification_report.txt",
    OUTPUT_DIR / "predictions.csv",
    OUTPUT_DIR / "logistic_regression_model.joblib",
    VISUALIZATION_DIR / "confusion_matrix.png",
    VISUALIZATION_DIR / "roc_curve.png"
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

if all_outputs_exist:
    print("LOGISTIC REGRESSION TASK COMPLETED SUCCESSFULLY!")
else:
    print("Some output files are missing. Check the errors above.")

print("\nOutput directory:")
print(OUTPUT_DIR)