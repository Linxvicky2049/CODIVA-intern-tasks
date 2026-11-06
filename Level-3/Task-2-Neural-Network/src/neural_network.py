from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
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
print("CODVEDA INTERNSHIP: NEURAL NETWORK CLASSIFICATION")
print("=" * 60)

# ==================================================
# 2. LOAD DATA
# ==================================================

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError("Churn dataset files could not be found.")

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

if "Churn" not in train_df.columns or "Churn" not in test_df.columns:
    raise ValueError("Churn target column is missing.")

feature_columns = [
    column for column in train_df.columns
    if column != "Churn"
]

X_train = train_df[feature_columns].copy()
y_train = train_df["Churn"].astype(bool)

X_test = test_df[feature_columns].copy()
y_test = test_df["Churn"].astype(bool)

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

# ==================================================
# 6. BUILD NEURAL NETWORK
# ==================================================

neural_network = MLPClassifier(
    random_state=42,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", neural_network)
])

# ==================================================
# 7. HYPERPARAMETER TUNING
# ==================================================

parameter_grid = {
    "classifier__hidden_layer_sizes": [
        (50,),
        (100,),
        (100, 50)
    ],
    "classifier__activation": [
        "relu",
        "tanh"
    ],
    "classifier__alpha": [
        0.0001,
        0.001
    ]
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

print("\nTRAINING NEURAL NETWORK...")
print("Performing five-fold cross-validation.")
print("This may take several minutes.")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBEST PARAMETERS")
print(grid_search.best_params_)

print(
    "Best cross-validation F1:",
    round(grid_search.best_score_, 4)
)

# ==================================================
# 8. PREDICTIONS
# ==================================================

y_pred = best_model.predict(X_test)

classifier = best_model.named_steps["classifier"]

classes = list(classifier.classes_)

if True not in classes:
    raise ValueError("Positive churn class was not found.")

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

roc_auc = roc_auc_score(y_test, y_probability)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\nMODEL PERFORMANCE")
print("=" * 40)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nCLASSIFICATION REPORT")
print(report)

# ==================================================
# 10. SAVE METRICS
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

    file.write("NEURAL NETWORK CLASSIFICATION REPORT\n")
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
    OUTPUT_DIR / "neural_network_model.joblib"
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

plt.title("Neural Network Confusion Matrix")
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

plt.title("Neural Network ROC Curve")
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "roc_curve.png",
    dpi=300
)

plt.close()

# ==================================================
# 16. TRAINING LOSS CURVE
# ==================================================

loss_curve = classifier.loss_curve_

loss_df = pd.DataFrame({
    "Iteration": range(1, len(loss_curve) + 1),
    "Training_Loss": loss_curve
})

loss_df.to_csv(
    OUTPUT_DIR / "training_loss.csv",
    index=False
)

plt.figure(figsize=(10, 6))

plt.plot(
    loss_curve,
    label="Training Loss"
)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Neural Network Training Loss Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "training_loss_curve.png",
    dpi=300
)

plt.close()

# ==================================================
# 17. SAVE MODEL SUMMARY
# ==================================================

summary = f"""
NEURAL NETWORK CLASSIFICATION SUMMARY

Training records: {len(X_train)}
Testing records: {len(X_test)}

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}
ROC-AUC: {roc_auc:.4f}

Cross-validation F1: {grid_search.best_score_:.4f}

Training iterations: {classifier.n_iter_}

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
# 18. OUTPUT VALIDATION
# ==================================================

expected_files = [
    OUTPUT_DIR / "evaluation_metrics.csv",
    OUTPUT_DIR / "classification_report.txt",
    OUTPUT_DIR / "predictions.csv",
    OUTPUT_DIR / "neural_network_model.joblib",
    OUTPUT_DIR / "training_loss.csv",
    OUTPUT_DIR / "model_summary.txt",
    VISUALIZATION_DIR / "confusion_matrix.png",
    VISUALIZATION_DIR / "roc_curve.png",
    VISUALIZATION_DIR / "training_loss_curve.png"
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
    print("NEURAL NETWORK TASK COMPLETED SUCCESSFULLY!")
else:
    print("Some output files are missing.")
    raise RuntimeError("Output validation failed.")

print("\nOutput directory:")
print(OUTPUT_DIR)