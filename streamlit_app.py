# streamlit_app.py

import streamlit as st
import pandas as pd
from pipeline import load_and_preprocess, make_lagged_features, train_and_save_model, load_model, predict_next_year

st.set_page_config(page_title="Malaria Incidence Forecasting", layout="centered")

st.title("🦟 Malaria Incidence Forecasting (Bhutan)")

# --- Load / Preprocess Data ---
if st.button("Load & Preprocess Data"):
    try:
        df = load_and_preprocess()
        st.success("Data processed successfully!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error during data loading / preprocessing: {e}")

# --- Train Model ---
if st.button("Train Model"):
    try:
        df = load_and_preprocess()
        df_model = make_lagged_features(df, lags=2)
        results = train_and_save_model(df_model)

        st.success("Model trained and saved successfully!")
        st.write("**Model Performance on Test Set:**")
        st.write(f"MSE: {results['mse']:.4f}")
        st.write(f"RMSE: {results['rmse']:.4f}")
        st.write(f"R²: {results['r2']:.4f}")
    except Exception as e:
        st.error(f"Error during model training: {e}")

# --- Predict Next Year ---
if st.button("Predict Next Year’s Incidence"):
    try:
        # Preprocess & generate lag features
        df = load_and_preprocess()
        df_model = make_lagged_features(df, lags=2)

        # Load the trained model
        model = load_model()

        # Make prediction
        pred = predict_next_year(model, df_model)
        st.success(f"🧮 Predicted Malaria Incidence for Next Year: **{pred:.4f}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --- Show Processed Data Table ---
if st.checkbox("Show Full Processed Data"):
    try:
        df = load_and_preprocess()
        st.dataframe(df)
    except Exception as e:
        st.error(f"Cannot load processed data: {e}")
