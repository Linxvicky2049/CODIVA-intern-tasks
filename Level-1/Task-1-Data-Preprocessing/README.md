# Stock Market Data Preprocessing

### Codveda Technologies | Machine Learning Internship

**Level:** 1  
**Task:** 1 - Data Preprocessing  
**Dataset:** Stock Prices Data Set  
**Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## 1. Project Overview

This project demonstrates a complete data preprocessing workflow using the Stock Prices Data Set provided for the Codveda Technologies Machine Learning Internship.

The objective is to transform raw stock market data into a clean, structured, and machine-learning-ready dataset.

The project covers data inspection, cleaning, missing-value treatment, feature engineering, categorical encoding, numerical scaling, chronological data splitting, and visualization.

The processed dataset is prepared for potential use in subsequent machine learning applications.

## 2. Project Objectives

The main objectives are:

- Inspect the structure and quality of the raw dataset.
- Identify and handle missing values.
- Detect and remove duplicate records.
- Convert date values into the appropriate datetime format.
- Extract meaningful date-based features.
- Encode categorical stock symbols into numerical representations.
- Scale numerical features.
- Split the dataset chronologically into training and testing sets.
- Generate visualizations to communicate data quality and distributions.
- Save the processed datasets for future machine learning applications.

## 3. Dataset Description

The dataset contains historical stock market information from January 2014 to December 2017.

It includes the following columns:

| Column | Description |
|---|---|
| symbol | Stock ticker symbol |
| date | Trading date |
| open | Opening stock price |
| high | Highest stock price |
| low | Lowest stock price |
| close | Closing stock price |
| volume | Number of shares traded |

### Dataset Statistics

- Total records: 497,472
- Original features: 7
- Unique stock symbols: 505
- Date range: January 2, 2014 to December 29, 2017
- Duplicate records: 0
- Missing values: 27

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing pipelines |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SciPy | Sparse matrix storage |

## 5. Data Preprocessing Methodology

### 5.1 Data Loading and Inspection

The dataset was loaded using Pandas.

Initial inspection included:

- Dataset dimensions
- Column data types
- Missing-value counts
- Duplicate records
- Numerical statistics

### 5.2 Data Cleaning

The date column was converted into a proper datetime format.

Records containing invalid dates were removed, and duplicate records were eliminated.

The dataset was then sorted chronologically by date and stock symbol.

### 5.3 Missing Value Treatment

The dataset contained 27 missing numerical values:

| Feature | Missing Values |
|---|---:|
| open | 11 |
| high | 8 |
| low | 8 |
| Total | 27 |

Missing numerical values were replaced using median imputation through Scikit-learn's SimpleImputer.

Median imputation was selected because it is less sensitive to extreme values than mean imputation.

### 5.4 Feature Engineering

The original date column was used to extract three additional features:

- Year
- Month
- Day

The original date column was subsequently excluded from the model input features.

### 5.5 Categorical Encoding

The symbol column contains 505 unique stock symbols.

OneHotEncoder was used to convert these categorical values into numerical features.

The encoder was configured with `handle_unknown="ignore"` to handle previously unseen categories safely.

### 5.6 Numerical Feature Scaling

StandardScaler was applied to numerical features.

This transforms numerical values into standardized representations based on the training data's mean and standard deviation.

### 5.7 Preprocessing Pipeline

Scikit-learn Pipeline and ColumnTransformer were used to combine the numerical and categorical preprocessing operations.

The pipeline was fitted using training data only and subsequently applied to the testing data.

This prevents information from the testing dataset from influencing the fitted preprocessing parameters.

### 5.8 Chronological Train-Test Split

The dataset was split using unique trading dates.

- Training dataset: approximately 80%
- Testing dataset: approximately 20%

The split was performed chronologically rather than randomly because the dataset represents historical stock market observations.

This ensures that earlier observations are used for training and later observations are reserved for testing.

## 6. Results

### Original Dataset

| Metric | Result |
|---|---:|
| Records | 497,472 |
| Features | 7 |
| Missing values | 27 |
| Duplicate records | 0 |

### Processed Dataset

| Metric | Result |
|---|---:|
| Training records | 395,939 |
| Testing records | 101,533 |
| Training features | 508 |
| Testing features | 508 |
| Missing numerical values | Imputed |

### Training and Testing Periods

**Training:**
January 2, 2014 to March 14, 2017.

**Testing:**
March 15, 2017 to December 29, 2017.

### Feature Transformation

The final processed dataset contains 508 features:

- 8 numerical features.
- 500 one-hot encoded stock-symbol features.

The processed matrices are stored in sparse format to reduce storage requirements.

## 7. Visualizations

The project generates two visualizations.

### Missing Values Distribution

File: `outputs/visualizations/missing_values.png`

This chart displays the number of missing observations in each affected column before imputation.

### Stock Closing Price Distribution

File: `outputs/visualizations/closing_price_distribution.png`

This histogram illustrates the distribution of stock closing prices across the dataset.

## 8. Project Structure

```text
task-1/
│
├── data/
│   └── 2) Stock Prices Data Set.csv
│
├── outputs/
│   ├── X_train_processed.npz
│   ├── X_test_processed.npz
│   ├── feature_names.csv
│   ├── split_information.txt
│   │
│   └── visualizations/
│       ├── missing_values.png
│       └── closing_price_distribution.png
│
├── src/
│   └── preprocess.py
│
├── notebooks/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 9. Installation and Execution

### Prerequisites

- Python 3.10 or later
- Visual Studio Code
- Git

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project directory:

```bash
cd task-1
```

### Step 2: Create a Virtual Environment

```bash
python -m venv env
```

Activate it on Windows PowerShell:

```powershell
.\env\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add the Dataset

Place the provided Stock Prices Data Set CSV file inside:

```text
data/
```

### Step 5: Execute the Preprocessing Script

```bash
python src/preprocess.py
```

### Step 6: Inspect the Results

Processed datasets and visualizations will be generated inside:

```text
outputs/
```

## 10. Limitations and Future Improvements

Possible future improvements include:

- Additional time-series feature engineering.
- Technical indicators such as moving averages and volatility.
- More detailed statistical visualizations.
- Outlier detection and analysis.
- Stock-specific preprocessing strategies.
- Predictive modelling using regression or time-series algorithms.

This project focuses on data preparation and does not claim to train or evaluate a stock-price prediction model.

## 11. Conclusion

The project successfully demonstrates the transformation of a large historical stock market dataset into a structured format suitable for future machine learning workflows.

Missing numerical values were handled, categorical variables were encoded, numerical features were standardized, and the dataset was divided chronologically into training and testing sets.

The resulting datasets and visualizations provide a foundation for subsequent machine learning experiments.

---

**Developed as part of the Codveda Technologies Machine Learning Internship.**#   C O D I V A - i n t e r n - t a s k s  
 