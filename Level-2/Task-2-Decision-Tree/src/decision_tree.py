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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "Churn Prdiction Data" / "churn-bigml-80.csv"
TEST_PATH = PROJECT_ROOT / "data" / "Churn Prdiction Data" / "churn-bigml-20.csv"

OUTPUT_DIR = TASK_DIR / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CODVEDA INTERNSHIP: DECISION TREE CLASSIFICATION")
print("=" * 60)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError("Churn dataset files could not be found.")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nDATASET INFORMATION")
print("-" * 40)
print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

train_df = train_df.drop_duplicates().copy()
test_df = test_df.drop_duplicates().copy()

print("\nMISSING VALUES")
print("-" * 40)
print("Training missing values:", train_df.isnull().sum().sum())
print("Testing missing values:", test_df.isnull().sum().sum())

if "Churn" not in train_df.columns or "Churn" not in test_df.columns:
    raise ValueError("The Churn target column is missing.")

# Align testing columns to training columns.
feature_columns = [col for col in train_df.columns if col != "Churn"]

X_train = train_df[feature_columns].copy()
y_train = train_df["Churn"].copy()

X_test = test_df[feature_columns].copy()
y_test = test_df["Churn"].copy()

X_test = X_test.reindex(columns=feature_columns)

# --------------------------------------------------
# 4. IDENTIFY FEATURE TYPES
# --------------------------------------------------

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nFEATURE INFORMATION")
print("-" * 40)
print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

# --------------------------------------------------
# 5. PREPROCESSING
# --------------------------------------------------

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

# --------------------------------------------------
# 6. BUILD DECISION TREE MODEL
# --------------------------------------------------

decision_tree = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", decision_tree)
])

# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

print("\nTRAINING DECISION TREE MODEL...")
model.fit(X_train, y_train)

print("Model training completed successfully.")

# --------------------------------------------------
# 8. HYPERPARAMETER TUNING
# --------------------------------------------------

from sklearn.model_selection import GridSearchCV, StratifiedKFold

parameter_grid = {
    "classifier__max_depth": [3, 5, 7, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__criterion": ["gini", "entropy"]
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

print("\nPERFORMING HYPERPARAMETER TUNING...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBEST PARAMETERS")
print(grid_search.best_params_)
print("Best cross-validation F1:", round(grid_search.best_score_, 4))

# --------------------------------------------------
# 9. MODEL PREDICTIONS
# --------------------------------------------------

y_pred = best_model.predict(X_test)
y_probability = best_model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 10. MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

print("\nMODEL PERFORMANCE")
print("=" * 40)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

# --------------------------------------------------
# 11. SAVE EVALUATION RESULTS
# --------------------------------------------------

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Best Cross-Validation F1"
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

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

with open(OUTPUT_DIR / "classification_report.txt", "w") as file:
    file.write("DECISION TREE CLASSIFICATION REPORT\n")
    file.write("=" * 45 + "\n\n")
    file.write(report)
    file.write("\n\nBEST PARAMETERS\n")
    file.write(str(grid_search.best_params_))

# --------------------------------------------------
# 12. SAVE PREDICTIONS
# --------------------------------------------------

predictions = pd.DataFrame({
    "Actual_Churn": y_test.to_numpy(),
    "Predicted_Churn": y_pred,
    "Churn_Probability": y_probability
})

predictions.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)

# --------------------------------------------------
# 13. SAVE TRAINED MODEL
# --------------------------------------------------

joblib.dump(
    best_model,
    OUTPUT_DIR / "decision_tree_model.joblib"
)

# --------------------------------------------------
# 14. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

display.plot(cmap="Blues", values_format="d")
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 15. ROC CURVE
# --------------------------------------------------

RocCurveDisplay.from_predictions(
    y_test,
    y_probability
)

plt.title("Decision Tree ROC Curve")
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "roc_curve.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 16. FEATURE IMPORTANCE
# --------------------------------------------------

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
plt.title("Top 15 Decision Tree Feature Importances")
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "feature_importance.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 17. SAVE TREE VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(24, 12))

plot_tree(
    classifier,
    feature_names=feature_names,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Structure (First Three Levels)")
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "decision_tree_structure.png",
    dpi=200
)

plt.close()

# --------------------------------------------------
# 18. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("DECISION TREE TASK COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOUTPUT FILES:")
for file_path in sorted(OUTPUT_DIR.rglob("*")):
    if file_path.is_file():
        print(file_path.relative_to(TASK_DIR))

print("\nAll model outputs and visualizations saved successfully.")