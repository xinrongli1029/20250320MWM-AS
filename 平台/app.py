import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# **1️⃣ Set Streamlit Page**
st.set_page_config(page_title="As(III) Adsorption Capacity Prediction for MWM", layout="wide")

# **2️⃣ Page Header**
st.title("🔬 As(III) Adsorption Capacity Prediction for MWM")
st.markdown("This tool uses a **stacking ensemble model** to predict **As(III) adsorption capacity, q (mg/g)**, for `MWM` materials.")

# **3️⃣ Automatically Get Current Python File Path (for GitHub / Servers)**
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))  # Automatically adapts to the current file path


# **4️⃣ Load Models**
@st.cache_resource
def load_models():
    """Load base models, meta-model, and scaler"""
    try:
        base_models = joblib.load(os.path.join(MODEL_PATH, "base_models.pkl"))
        meta_model = joblib.load(os.path.join(MODEL_PATH, "meta_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
        st.sidebar.success("✅ Models loaded successfully!")
        return base_models, meta_model, scaler
    except FileNotFoundError as e:
        st.sidebar.error(f"❌ Model files not found! Error: {e}")
        return None, None, None

# **Load Models**
base_models, meta_model, scaler = load_models()

# **5️⃣ Ensure Models Are Loaded Correctly**
if base_models is None or meta_model is None or scaler is None:
    st.sidebar.error("⚠️ Model loading failed. Please check the paths and restart the program.")
    st.stop()

# **6️⃣ Real Feature Names**
feature_names = [
    "Initial concentration (mg/L)",
    "pH",
    "Dosage (g/L)",
    "Contact time (h)",
    "Illumination time (h)",
    "Silicate ion concentration (mg/L)",
    "Chloride ion concentration (mg/L)",
    "Fluoride ion concentration (mg/L)"
]

# **7️⃣ Move Feature Inputs to Sidebar**
st.sidebar.header("📊 Enter Feature Values")
input_data = []

for feature_name in feature_names:
    value = st.sidebar.number_input(f"{feature_name}", value=0.0, format="%.4f")
    input_data.append(value)

# **8️⃣ Prediction Button**
if st.sidebar.button("🚀 Predict"):
    try:
        # **Preprocess Data**
        X_input = np.array(input_data).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        # **Base Model Predictions**
        base_preds = [model.predict(X_scaled).reshape(-1, 1) for model in base_models.values()]
        X_meta = np.hstack(base_preds)

        # **Stacking Model Prediction**
        final_prediction = meta_model.predict(X_meta)[0]

        # **Display Results**
        st.success(f"📊 **Predicted As(III) adsorption capacity, q:** `{final_prediction:.4f} mg/g`")
    except Exception as e:
        st.sidebar.error(f"❌ Prediction failed! Error: {e}")
