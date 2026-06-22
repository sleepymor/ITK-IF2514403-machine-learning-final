import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Konfigurasi awal halaman
st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")

@st.cache_data
def load_dataset():
    return pd.read_csv("Streamlit/breast_cancer_raw.csv") 

@st.cache_resource
def load_models():
    scaler_obj = joblib.load("Streamlit/scaler.pkl")
    ml_models = {
        "Logistic Regression": joblib.load("Streamlit/model_LogReg_PCA.pkl"),
        "KNN": joblib.load("Streamlit/model_kNN_PCA.pkl"),
        "Random Forest": joblib.load("Streamlit/model_RF_PCA.pkl")
    }
    return scaler_obj, ml_models

# Load data dan model ke memory
df = load_dataset()
scaler, models = load_models()

# --- Setup Navigasi Sidebar ---
st.sidebar.title("Navigation")
menu_options = ["Home", "Dataset", "Visualization", "Prediction"]

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Bikin tombol sidebar secara dinamis
for option in menu_options:
    if st.sidebar.button(option, use_container_width=True):
        st.session_state.current_page = option

page = st.session_state.current_page

# --- Halaman Home ---
if page == "Home":
    st.title("Breast Cancer Prediction App")
    st.markdown("""
    Aplikasi ini memprediksi apakah tumor payudara termasuk:
    - **Benign** (Jinak)
    - **Malignant** (Ganas)

    Menggunakan model Machine Learning:
    - Logistic Regression
    - K-Nearest Neighbors
    - Random Forest
    """)
    
    st.subheader("Dataset Shape")
    st.write(f"Total baris dan kolom: {df.shape}")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# --- Halaman Dataset ---
elif page == "Dataset":
    st.title("Dataset")
    st.dataframe(df)
    
    st.subheader("Statistics")
    st.dataframe(df.describe())

# --- Halaman Visualisasi ---
elif page == "Visualization":
    st.title("Visualization")
    
    st.subheader("Diagnosis Distribution")
    diag_counts = df["diagnosis"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(["Benign", "Malignant"], diag_counts.values, color=['#4CAF50', '#F44336'])
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes(include=[np.number]) 
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    im = ax2.imshow(num_df.corr(), cmap="coolwarm")
    plt.colorbar(im)
    st.pyplot(fig2)

# --- Halaman Prediksi ---
elif page == "Prediction":
    st.title("Prediction")
    st.write("Silakan masukkan nilai pengukuran tumor di bawah ini.")

    selected_model = st.segmented_control(
        "Choose Model",
        options=["Logistic Regression", "KNN", "Random Forest"],
        default="Logistic Regression"
    )
    
    active_model = models[selected_model]

    # Layout form input pakai 2 kolom
    left_col, right_col = st.columns(2)
    
    with left_col:
        rad_mean = st.number_input("Radius Mean", min_value=0.0, value=14.0)
        tex_mean = st.number_input("Texture Mean", min_value=0.0, value=19.0)
        per_mean = st.number_input("Perimeter Mean", min_value=0.0, value=90.0)
        area_mean = st.number_input("Area Mean", min_value=0.0, value=600.0)
        smooth_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1)

    with right_col:
        comp_mean = st.number_input("Compactness Mean", min_value=0.0, value=0.1)
        conc_mean = st.number_input("Concavity Mean", min_value=0.0, value=0.1)
        conc_pts_mean = st.number_input("Concave Points Mean", min_value=0.0, value=0.05)
        sym_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.2)
        frac_dim_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, value=0.06)

    # Eksekusi prediksi
    if st.button("Predict", type="primary"):
        # Fitur dummy (index 10-29) untuk melengkapi bentuk matriks
        dummy_vals = [0] * 20
        
        # Ekstraksi feature engineering tambahan
        area_perim_ratio = area_mean / per_mean if per_mean != 0 else 0
        rad_growth = 0 
        conc_score = conc_mean + conc_pts_mean
        
        # Pengelompokan kategori radius 
        rad_cat = 0 if rad_mean < 12 else (1 if rad_mean < 18 else 2)

        # Susun array 34 fitur
        X_new = np.array([[
            rad_mean, tex_mean, per_mean, area_mean, smooth_mean,
            comp_mean, conc_mean, conc_pts_mean, sym_mean, frac_dim_mean,
            *dummy_vals, 
            area_perim_ratio, rad_growth, conc_score, rad_cat
        ]])

        # Standarisasi data input
        X_scaled = scaler.transform(X_new)

        # Hasil prediksi (pipeline menangani proses PCA)
        pred_class = active_model.predict(X_scaled)[0]
        pred_proba = active_model.predict_proba(X_scaled)[0]

        # Tampilkan output ke user
        st.divider()
        st.subheader("Prediction Result")
        
        if pred_class == 1:
            st.error("Malignant Tumor Detected")
        else:
            st.success("Benign Tumor Detected")

        st.write(f"**Probability:** Benign ({pred_proba[0]:.1%}) | Malignant ({pred_proba[1]:.1%})")
        st.caption(f"Model used: {selected_model}")
