import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --- Configuration ---
PROCESSED_FILE_PATH = Path("data/processed/processed_malaria_data.csv")
MODEL_DIR = Path("models")
MODEL_FILE_PATH = MODEL_DIR / "malaria_incidence_forecast_model.pkl"
TARGET_INDICATOR = 'MALARIA_EST_INCIDENCE' # We will forecast the estimated incidence

# --- Step 4: Feature Engineering ---

def create_lagged_features(df, target_col, lags=3):
    """
    Creates lagged (time-shifted) features for time-series forecasting.
    A lag of 1 means using last year's data to predict this year.
    """
    print(f"\n--- Step 4: Feature Engineering (Creating Lags up to {lags} Years) ---")
    
    # Set the year as index for clean shifting
    df = df.set_index('year').copy()
    
    # Drop rows where the target column is missing, though preprocessing should have handled this
    df.dropna(subset=[target_col], inplace=True) 

    # Create lagged features for ALL indicators
    for col in df.columns:
        for lag in range(1, lags + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
    # Define the forecast target (y)
    df['forecast_target'] = df[target_col]

    # Drop the original, non-lagged features (X) and the NaN rows created by shifting
    # We only keep the lagged versions for training (X) and the current target (y).
    df_features = df.drop(columns=df.columns[:len(df.columns) - lags * len(df.columns.drop('forecast_target')) - 1])
    
    df_features.dropna(inplace=True)
    
    print(f"Total features created (including target): {df_features.shape[1]}")
    print(f"Final data shape after dropping NaNs due to lagging: {df_features.shape}")
    
    return df_features

# --- Step 5: Machine Learning Modeling ---

def train_and_evaluate_model(df, target_col='forecast_target'):
    """Trains a Linear Regression model for patient load forecasting (as a proxy)."""
    print("\n--- Step 5: Machine Learning Modeling (Linear Regression) ---")
    
    # Define X (features) and y (target)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # Time series data must not be shuffled

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    # Train the candidate model (Linear Regression for forecasting)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Metrics:")
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
        # Load the processed data
        processed_df = pd.read_csv(PROCESSED_FILE_PATH)
        
        # 4. Feature Engineering
        lagged_df = create_lagged_features(processed_df, TARGET_INDICATOR, lags=2) # Use lag 2 for stability

        # 5. Machine Learning Modeling
        if not lagged_df.empty:
            train_and_evaluate_model(lagged_df)
        else:
            print("Error: Lagged dataframe is empty. Cannot proceed to training.")

    except FileNotFoundError:
        print(f"Error: Processed data not found at {PROCESSED_FILE_PATH}. Please run data_loader.py (or your Streamlit app) first.")
    except Exception as e:
        print(f"An unexpected error occurred during ML pipeline execution: {e}")
