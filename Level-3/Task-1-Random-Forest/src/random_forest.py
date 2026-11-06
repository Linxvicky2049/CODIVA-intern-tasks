from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")

# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "Churn Prdiction Data"
    / "churn-bigml-80.csv"
)

TEST_PATH = (
    PROJECT_ROOT / "data" / "Churn Prdiction Data"
    / "churn-bigml-20.csv"
)

OUTPUT_DIR = TASK_DIR / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CODVEDA INTERNSHIP: RANDOM FOREST CLASSIFICATION")
print("=" * 60)

# ==================================================
# 2. LOAD DATA
# ==================================================

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training dataset not found: {TRAIN_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Testing dataset not found: {TEST_PATH}")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nDATASET INFORMATION")
print("-" * 40)

print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

# ==================================================
# 3. DATA CLEANING
# ==================================================

train_df = train_df.drop_duplicates().copy()
test_df = test_df.drop_duplicates().copy()

print("\nDATA QUALITY")
print("-" * 40)

print("Training missing values:", train_df.isnull().sum().sum())
print("Testing missing values:", test_df.isnull().sum().sum())

if "Churn" not in train_df.columns:
    raise ValueError("Churn target column missing from training data.")

if "Churn" not in test_df.columns:
    raise ValueError("Churn target column missing from testing data.")

feature_columns = [
    column for column in train_df.columns
    if column != "Churn"
]

X_train = train_df[feature_columns].copy()
y_train = train_df["Churn"].copy()

X_test = test_df[feature_columns].copy()
y_test = test_df["Churn"].copy()

X_test = X_test.reindex(columns=feature_columns)

# ==================================================
# 4. IDENTIFY FEATURE TYPES
# ==================================================

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nFEATURE INFORMATION")
print("-" * 40)

print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

# ==================================================
# 5. PREPROCESSING
# ==================================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ==================================================
# 6. BUILD RANDOM FOREST PIPELINE
# ==================================================

random_forest = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", random_forest)
])

# ==================================================
# 7. HYPERPARAMETER TUNING
# ==================================================

parameter_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameter_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("\nTRAINING AND TUNING RANDOM FOREST...")
print("This may take a few minutes.")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBEST PARAMETERS")
print(grid_search.best_params_)

print(
    "Best cross-validation F1:",
    round(grid_search.best_score_, 4)
)

# ==================================================
# 8. MODEL PREDICTIONS
# ==================================================

y_pred = best_model.predict(X_test)

classes = list(best_model.named_steps["classifier"].classes_)

if True not in classes:
    raise ValueError("Expected positive churn class True was not found.")

positive_index = classes.index(True)

y_probability = best_model.predict_proba(X_test)[:, positive_index]

# ==================================================
# 9. MODEL EVALUATION
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, pos_label=True, zero_division=0
)

recall = recall_score(
    y_test, y_pred, pos_label=True, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, pos_label=True, zero_division=0
)

roc_auc = roc_auc_score(
    y_test.astype(bool),
    y_probability
)

print("\nMODEL PERFORMANCE")
print("=" * 40)

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

print("\nCLASSIFICATION REPORT")
print(report)

# ==================================================
# 10. SAVE EVALUATION METRICS
# ==================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Cross-Validation F1"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        grid_search.best_score_
    ]
})

metrics.to_csv(
    OUTPUT_DIR / "evaluation_metrics.csv",
    index=False
)

# ==================================================
# 11. SAVE CLASSIFICATION REPORT
# ==================================================

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("RANDOM FOREST CLASSIFICATION REPORT\n")
    file.write("=" * 45 + "\n\n")
    file.write(report)

    file.write("\n\nBEST PARAMETERS\n")
    file.write(str(grid_search.best_params_))

# ==================================================
# 12. SAVE PREDICTIONS
# ==================================================

predictions = pd.DataFrame({
    "Actual_Churn": y_test.to_numpy(),
    "Predicted_Churn": y_pred,
    "Churn_Probability": y_probability
})

predictions.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)

# ==================================================
# 13. SAVE TRAINED MODEL
# ==================================================

joblib.dump(
    best_model,
    OUTPUT_DIR / "random_forest_model.joblib"
)

# ==================================================
# 14. CONFUSION MATRIX
# ==================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues",
    values_format="d"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

# ==================================================
# 15. ROC CURVE
# ==================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_probability,
    pos_label=True
)

plt.title("Random Forest ROC Curve")
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "roc_curve.png",
    dpi=300
)

plt.close()

# ==================================================
# 16. FEATURE IMPORTANCE
# ==================================================

classifier = best_model.named_steps["classifier"]

fitted_preprocessor = best_model.named_steps["preprocessor"]

feature_names = fitted_preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

top_features = importance_df.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 15 Random Forest Feature Importances")

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "feature_importance.png",
    dpi=300
)

plt.close()

# ==================================================
# 17. SAVE FINAL SUMMARY
# ==================================================

summary = f"""
RANDOM FOREST CLASSIFICATION SUMMARY

Training records: {len(X_train)}
Testing records: {len(X_test)}

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}
ROC-AUC: {roc_auc:.4f}

Cross-validation F1: {grid_search.best_score_:.4f}

Best Parameters:
{grid_search.best_params_}
"""

with open(
    OUTPUT_DIR / "model_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)

# ==================================================
# 18. FINAL VALIDATION
# ==================================================

expected_files = [
    OUTPUT_DIR / "evaluation_metrics.csv",
    OUTPUT_DIR / "classification_report.txt",
    OUTPUT_DIR / "predictions.csv",
    OUTPUT_DIR / "feature_importance.csv",
    OUTPUT_DIR / "random_forest_model.joblib",
    OUTPUT_DIR / "model_summary.txt",
    VISUALIZATION_DIR / "confusion_matrix.png",
    VISUALIZATION_DIR / "roc_curve.png",
    VISUALIZATION_DIR / "feature_importance.png"
]

print("\n========== OUTPUT VALIDATION ==========")

all_created = True

for file_path in expected_files:

    exists = file_path.exists()

    print(
        f"{file_path.name}:",
        "CREATED" if exists else "MISSING"
    )

    if not exists:
        all_created = False

print("\n========== FINAL RESULT ==========")

if all_created:
    print("All output files created: True")
    print("RANDOM FOREST TASK COMPLETED SUCCESSFULLY!")
else:
    print("Some output files are missing.")
    raise RuntimeError("Output validation failed.")

print("\nOutput directory:")
print(OUTPUT_DIR)