import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# --- Configuration ---
RAW_DATA_PATH = "malaria_indicators_btn.csv"
PROCESSED_DIR = Path("data/processed")
PROCESSED_FILE_PATH = PROCESSED_DIR / "processed_malaria_data.csv"
MODEL_DIR = Path("models")
MODEL_FILE_PATH = MODEL_DIR / "malaria_incidence_forecast_model.pkl"
TARGET_INDICATOR = 'MALARIA_EST_INCIDENCE' # We will forecast the estimated incidence

# --- Re-using Data Loading/Preprocessing Functions from data_loader.py for robustness ---

def load_and_preprocess_if_missing(file_path):
    """
    Checks if the processed file exists. If not, it runs the full preprocessing pipeline 
    from the raw data and saves the result.
    """
    if PROCESSED_FILE_PATH.exists():
        print(f"Loading existing processed data from: {PROCESSED_FILE_PATH}")
        return pd.read_csv(PROCESSED_FILE_PATH)

    print(f"Processed file not found. Running full preprocessing pipeline...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(RAW_DATA_PATH, header=1)
    except FileNotFoundError:
        print(f"Critical Error: Raw data file '{RAW_DATA_PATH}' not found.")
        return None
    
    # Initial Cleaning and Validation (Simplified)
    df.columns = df.columns.str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip().str.replace(r'\s+', '_', regex=True)
    df = df.rename(columns={'gho_display': 'indicator_name', 'year_display': 'year', 'numeric': 'value_num'})
    
    cols_to_drop = [col for col in df.columns if any(keyword in col for keyword in ['code', 'url', 'startyear', 'endyear', 'region', 'dimension', 'low', 'high', 'country'])]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df = df.rename(columns={'value_num': 'malaria_cases_or_rate'})
    df = df[['indicator_name', 'year', 'malaria_cases_or_rate']].copy()

    # 2. Preprocessing Steps (Imputation, Outlier Handling, Pivoting, Scaling)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    df['malaria_cases_or_rate'] = imputer.fit_transform(df[['malaria_cases_or_rate']])

    # Outlier handling (Winsorization)
    Q1 = df['malaria_cases_or_rate'].quantile(0.25)
    Q3 = df['malaria_cases_or_rate'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['malaria_cases_or_rate'] = np.clip(df['malaria_cases_or_rate'], lower_bound, upper_bound)
    
    # Pivoting
    df_pivot = df.pivot_table(index='year', columns='indicator_name', values='malaria_cases_or_rate', aggfunc='first').reset_index()
    df_pivot = df_pivot.fillna(method='ffill').fillna(method='bfill')
    
    # Scaling
    features_to_scale = df_pivot.columns.drop('year').tolist()
    scaler = StandardScaler()
    df_pivot[features_to_scale] = scaler.fit_transform(df_pivot[features_to_scale])
    df_pivot = df_pivot.rename(columns=lambda x: x.replace(' ', '_')) 

    # Save and return
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_pivot.to_csv(PROCESSED_FILE_PATH, index=False)
    print(f"Successfully generated and stored processed data to: {PROCESSED_FILE_PATH}")
    
    return df_pivot


# --- Step 4: Feature Engineering ---

def create_lagged_features(df, target_col, lags=2): # Reduced lag to 2 for better stability on small datasets
    """
    Creates lagged (time-shifted) features for time-series forecasting.
    """
    print(f"\n--- Step 4: Feature Engineering (Creating Lags up to {lags} Years) ---")
    
    df = df.set_index('year').copy()
    
    # Ensure the target exists before dropping NaNs
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in the processed data.")
        return pd.DataFrame() # Return empty DataFrame

    df.dropna(subset=[target_col], inplace=True) 

    # Create lagged features for ALL indicators
    for col in df.columns:
        for lag in range(1, lags + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
    df['forecast_target'] = df[target_col]

    # Drop the original, non-lagged features
    df_features = df.drop(columns=[col for col in df.columns if not ('lag' in col or col == 'forecast_target')])
    
    df_features.dropna(inplace=True)
    
    print(f"Total features created (including target): {df_features.shape[1]}")
    print(f"Training/Testing rows available: {df_features.shape[0]}")
    
    return df_features

# --- Step 5: Machine Learning Modeling ---

def train_and_evaluate_model(df, target_col='forecast_target'):
    """Trains a Linear Regression model for forecasting."""
    print("\n--- Step 5: Machine Learning Modeling (Linear Regression) ---")
    
    if df.shape[0] < 5:
        print("Error: Not enough data points to reliably train and test (need at least 5 rows).")
        return None

    # Define X (features) and y (target)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data (shuffle=False for time series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Metrics (on test set):")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    # Save the final model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE_PATH)
    print(f"\nModel successfully saved to: {MODEL_FILE_PATH}")

    return model

# --- Main Execution ---

if __name__ == "__main__":
    try:
        # Load or generate the processed data
        processed_df = load_and_preprocess_if_missing(PROCESSED_FILE_PATH)
        
        if processed_df is not None and not processed_df.empty:
            # 4. Feature Engineering
            lagged_df = create_lagged_features(processed_df, TARGET_INDICATOR, lags=2) 

            # 5. Machine Learning Modeling
            if not lagged_df.empty:
                train_and_evaluate_model(lagged_df)
            else:
                print("Error: Lagged dataframe is empty. Cannot proceed to training.")
        else:
            print("ML pipeline aborted due to data loading/generation failure.")

    except Exception as e:
        print(f"An unexpected error occurred during ML pipeline execution: {e}")
