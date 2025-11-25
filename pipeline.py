# pipeline.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# --- Paths and Config ---
RAW_DATA_PATH = Path("data") / "malaria_indicators_btn.csv"
PROCESSED_DIR = Path("data_processed")
PROCESSED_FILE_PATH = PROCESSED_DIR / "processed_malaria_data.csv"

MODEL_DIR = Path("models")
MODEL_FILE_PATH = MODEL_DIR / "malaria_incidence_model.pkl"

TARGET_INDICATOR = "malaria_incidence"


# --- Step 1 & 2: Load + Preprocess ---
def load_and_preprocess():
    """
    Load raw data, clean, impute, pivot, scale, and return a processed DataFrame.
    """
    # If processed data already exists, load it
    if PROCESSED_FILE_PATH.exists():
        df = pd.read_csv(PROCESSED_FILE_PATH)
        return df

    # Otherwise, load raw
    df = pd.read_csv(RAW_DATA_PATH, header=1)

    # Clean column names
    df.columns = (
        df.columns.str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Rename key columns
    df = df.rename(columns={
        "gho_display": "indicator_name",
        "year_display": "year",
        "numeric": "value_num",
    })

    # Drop unnecessary columns
    drop_keywords = ["code", "url", "startyear", "endyear", "region", "dimension", "low", "high", "country"]
    cols_to_drop = [col for col in df.columns if any(k in col for k in drop_keywords)]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Keep only relevant fields
    df = df[["indicator_name", "year", "value_num"]].copy()
    df = df.rename(columns={"value_num": "malaria_cases_or_rate"})

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    df["malaria_cases_or_rate"] = imputer.fit_transform(df[["malaria_cases_or_rate"]])

    # Outlier handling with IQR winsorization
    Q1 = df["malaria_cases_or_rate"].quantile(0.25)
    Q3 = df["malaria_cases_or_rate"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df["malaria_cases_or_rate"] = np.clip(df["malaria_cases_or_rate"], lower, upper)

    # Pivot: years × indicators
    df_pivot = (
        df.pivot_table(
            index="year",
            columns="indicator_name",
            values="malaria_cases_or_rate",
            aggfunc="first"
        )
        .reset_index()
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    # Clean pivoted column names
    df_pivot.columns = (
        df_pivot.columns.str.lower()
        .str.replace(r"[^a-z0-9_]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )

    # Rename incidence-like column
    incidence_cols = [c for c in df_pivot.columns if "incidence" in c]
    if incidence_cols:
        df_pivot = df_pivot.rename(columns={incidence_cols[0]: TARGET_INDICATOR})

    # Scale features (excluding year)
    scaler = StandardScaler()
    features = [c for c in df_pivot.columns if c != "year"]
    df_pivot[features] = scaler.fit_transform(df_pivot[features])

    # Save processed data
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_pivot.to_csv(PROCESSED_FILE_PATH, index=False)

    return df_pivot


# --- Step 3: Create Lagged Features ---
def make_lagged_features(df, lags=2):
    """
    Given processed DataFrame, create lagged features and target for forecasting.
    """
    df = df.set_index("year").copy()

    # Create lag features
    for col in df.columns:
        for lag in range(1, lags + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Target: next-year incidence
    df["forecast_target"] = df[TARGET_INDICATOR]

    # Keep only lagged features + target
    lagged_cols = [c for c in df.columns if ("lag" in c)]
    df_model = df[lagged_cols + ["forecast_target"]].dropna()

    return df_model


# --- Step 4: Train Model ---
def train_and_save_model(df_model):
    """
    Train a Linear Regression model on lagged features, save it, and return metrics.
    """
    X = df_model.drop(columns=["forecast_target"])
    y = df_model["forecast_target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Save the model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE_PATH)

    return {
        "model": model,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


# --- Step 5: Load & Predict ---
def load_model():
    """Load and return the trained model."""
    if not MODEL_FILE_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_FILE_PATH}")
    model = joblib.load(MODEL_FILE_PATH)
    return model


def predict_next_year(model, df_model):
    """
    Predict the next year's target, based on the most recent lagged row.
    """
    # Use the last row of features only (drop target)
    X_latest = df_model.drop(columns=["forecast_target"]).iloc[[-1]]
    prediction = model.predict(X_latest)[0]
    return float(prediction)
