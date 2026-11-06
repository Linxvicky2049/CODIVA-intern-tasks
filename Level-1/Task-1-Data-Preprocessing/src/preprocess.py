
"""
CODVEDA TECHNOLOGIES
Machine Learning Internship
Level 1 - Task 1: Data Preprocessing

Dataset: Stock Prices Data Set

Operations:
1. Data loading and inspection
2. Data cleaning
3. Missing value handling
4. Date feature extraction
5. Categorical encoding
6. Numerical feature scaling
7. Chronological train-test splitting
8. Saving processed datasets
9. Generating visualizations
"""
import os
import numpy as np
import pandas as pd

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import save_npz

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

# Automatically locate the main internship repository.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Locate this specific task folder.
TASK_DIR = Path(__file__).resolve().parents[1]

# Dataset location.
DATA_PATH = PROJECT_ROOT / "data" / "2) Stock Prices Data Set.csv"

# Task-specific output locations.
OUTPUT_DIR = TASK_DIR / "outputs"

VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# Create directories if they do not exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Dataset:", DATA_PATH)
print("Output directory:", OUTPUT_DIR)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Stock dataset not found: {DATA_PATH}"
    )

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\n========== STOCK DATA PREPROCESSING ==========")

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 3. INITIAL DATA INSPECTION
# --------------------------------------------------

print("\n========== INITIAL INSPECTION ==========")

print("\nDataset information:")
df.info()

print("\nMissing values before preprocessing:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())


# --------------------------------------------------
# 4. DATA CLEANING
# --------------------------------------------------

print("\n========== DATA CLEANING ==========")

# Convert date column to datetime
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# Remove records with invalid dates
df = df.dropna(subset=["date"])

# Remove duplicate records
df = df.drop_duplicates()

# Sort records chronologically
df = df.sort_values(
    by=["date", "symbol"]
).reset_index(drop=True)

print("Date conversion completed.")
print("Duplicate removal completed.")
print("Chronological sorting completed.")

print("Cleaned dataset shape:", df.shape)


# --------------------------------------------------
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------

print("\n========== TRAIN-TEST SPLITTING ==========")

# Split using unique dates so records from the
# same trading day cannot appear in both sets.

unique_dates = sorted(df["date"].unique())

split_index = int(len(unique_dates) * 0.80)

split_date = pd.Timestamp(
    unique_dates[split_index]
)

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()

print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print(
    "Training period:",
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)

print(
    "Testing period:",
    test_df["date"].min(),
    "to",
    test_df["date"].max()
)


# --------------------------------------------------
# 6. FEATURE ENGINEERING
# --------------------------------------------------

print("\n========== FEATURE ENGINEERING ==========")

def create_features(data):
    """Extract useful features from the date column."""

    data = data.copy()

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day

    return data.drop(columns=["date"])


X_train = create_features(train_df)
X_test = create_features(test_df)

print("Year, month, and day features created.")


# --------------------------------------------------
# 7. DEFINE FEATURE COLUMNS
# --------------------------------------------------

print("\n========== FEATURE PREPARATION ==========")

categorical_features = ["symbol"]

numerical_features = [
    column
    for column in X_train.select_dtypes(
        include=["number"]
    ).columns
]

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# --------------------------------------------------
# 8. BUILD PREPROCESSING PIPELINE
# --------------------------------------------------

print("\n========== BUILDING PREPROCESSING PIPELINE ==========")

# Numerical preprocessing:
# 1. Replace missing values with median.
# 2. Standardize numerical values.

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing:
# 1. Replace missing values with most frequent value.
# 2. Convert categories into numerical columns.

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


# Combine both pipelines.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 9. FIT AND TRANSFORM DATA
# --------------------------------------------------

print("\n========== TRANSFORMING DATA ==========")

# Fit the preprocessing pipeline on training data only.
X_train_processed = preprocessor.fit_transform(
    X_train
)

# Apply the fitted pipeline to testing data.
X_test_processed = preprocessor.transform(
    X_test
)

print("Training preprocessing completed.")
print("Testing preprocessing completed.")

print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# --------------------------------------------------
# 10. SAVE PROCESSED DATA
# --------------------------------------------------

print("\n========== SAVING PROCESSED DATA ==========")

save_npz(
    os.path.join(
        OUTPUT_DIR,
        "X_train_processed.npz"
    ),
    X_train_processed
)

save_npz(
    os.path.join(
        OUTPUT_DIR,
        "X_test_processed.npz"
    ),
    X_test_processed
)


# Save feature names.

feature_names = preprocessor.get_feature_names_out()

pd.Series(feature_names).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_names.csv"
    ),
    index=False,
    header=["feature_name"]
)


# Save split information.

with open(
    os.path.join(
        OUTPUT_DIR,
        "split_information.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"Training records: {len(X_train)}\n"
    )

    file.write(
        f"Testing records: {len(X_test)}\n"
    )

    file.write(
        f"Training period: {train_df['date'].min()} "
        f"to {train_df['date'].max()}\n"
    )

    file.write(
        f"Testing period: {test_df['date'].min()} "
        f"to {test_df['date'].max()}\n"
    )

print("Processed datasets saved successfully.")


# --------------------------------------------------
# 11. VISUALIZATION: MISSING VALUES
# --------------------------------------------------

print("\n========== GENERATING VISUALIZATIONS ==========")

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

plt.figure(figsize=(10, 6))

sns.barplot(
    x=missing_values.index,
    y=missing_values.values
)

plt.title("Missing Values Before Preprocessing")
plt.xlabel("Dataset Columns")
plt.ylabel("Number of Missing Values")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "missing_values.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 12. VISUALIZATION: STOCK PRICE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    df["close"],
    bins=50,
    kde=True
)

plt.title("Distribution of Stock Closing Prices")
plt.xlabel("Closing Price")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "closing_price_distribution.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 13. FINAL PREPROCESSING REPORT
# --------------------------------------------------

print("\n========== PREPROCESSING SUMMARY ==========")

print("Original records:", len(pd.read_csv(DATA_PATH)))

print("Cleaned records:", len(df))

print("Training records:", len(X_train))

print("Testing records:", len(X_test))

print(
    "\nMissing values before imputation:",
    df.isnull().sum().sum()
)

print("\n========== PROCESSED DATA QUALITY ==========")

print(
    "All training values are finite:",
    np.isfinite(X_train_processed.data).all()
)

print(
    "All testing values are finite:",
    np.isfinite(X_test_processed.data).all()
)

print(
    "Processed features generated:",
    X_train_processed.shape[1]
)

print("\nPreprocessing completed successfully!")

print("\nGenerated files are available inside:")
print(OUTPUT_DIR)
