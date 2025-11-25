import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

RAW_DATA_PATH = "data/malaria_indicators_btn.csv"
PROCESSED_DIR = Path("data/processed")
PROCESSED_FILE_PATH = PROCESSED_DIR / "processed_malaria_data.csv"

MODEL_DIR = Path("models")
MODEL_FILE_PATH = MODEL_DIR / "malaria_incidence_forecast_model.pkl"

TARGET_INDICATOR = "malaria_incidence"


# -------------------------------------------------------------
# Load + Preprocess
# -------------------------------------------------------------
def load_and_preprocess_if_missing():
    if PROCESSED_FILE_PATH.exists():
        return pd.read_csv(PROCESSED_FILE_PATH)

    df = pd.read_csv(RAW_DATA_PATH, header=1)

    df.columns = (
        df.columns.str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    df = df.rename(
        columns={
            "gho_display": "indicator_name",
            "year_display": "year",
            "numeric": "value_num",
        }
    )

    df = df.rename(columns={"value_num": "malaria_cases_or_rate"})
    df = df[["indicator_name", "year", "malaria_cases_or_rate"]]

    df["malaria_cases_or_rate"] = SimpleImputer(strategy="median").fit_transform(
        df[["malaria_cases_or_rate"]]
    )

    Q1 = df["malaria_cases_or_rate"].quantile(0.25)
    Q3 = df["malaria_cases_or_rate"].quantile(0.75)
    IQR = Q3 - Q1
    df["malaria_cases_or_rate"] = np.clip(
        df["malaria_cases_or_rate"], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    )

    df_pivot = (
        df.pivot_table(
            index="year",
            columns="indicator_name",
            values="malaria_cases_or_rate",
            aggfunc="first",
        )
        .reset_index()
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    df_pivot.columns = (
        df_pivot.columns.str.lower()
        .str.replace(r"[^a-z0-9_]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )

    incidence_cols = [c for c in df_pivot.columns if "incidence" in c]
    if incidence_cols:
        df_pivot = df_pivot.rename(columns={incidence_cols[0]: TARGET_INDICATOR})

    scaler = StandardScaler()
    features = df_pivot.columns.drop("year")
    df_pivot[features] = scaler.fit_transform(df_pivot[features])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_pivot.to_csv(PROCESSED_FILE_PATH, index=False)

    return df_pivot


# -------------------------------------------------------------
# Lagged Features
# -------------------------------------------------------------
def create_lagged_features(df, lags=2):
    df = df.set_index("year").copy()

    for col in df.columns:
        for lag in range(1, lags + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    df["forecast_target"] = df[TARGET_INDICATOR]
    df = df.dropna()

    return df


# -------------------------------------------------------------
# Train + Evaluate
# -------------------------------------------------------------
def train_model(df):
    X = df.drop(columns=["forecast_target"])
    y = df["forecast_target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE_PATH)

    y_pred = model.predict(X_test)

    return {
        "model": model,
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }


# -------------------------------------------------------------
# Prediction
# -------------------------------------------------------------
def predict_next_year(model, df_lagged):
    latest = df_lagged.drop(columns=["forecast_target"]).iloc[-1:].values
    next_year_pred = model.predict(latest)[0]
    return next_year_pred
