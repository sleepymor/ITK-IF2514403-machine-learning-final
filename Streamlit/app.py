import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Perbaikan import matplotlib

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")

# --- DATA LOADER ---
@st.cache_data
def load_data():
    return pd.read_csv("breast_cancer_raw.csv") 

df = load_data()

@st.cache_resource
def load_components():
    scaler = jb.load("scaler.pkl")
    # Memuat model versi PCA Pipeline yang sudah kalian export
    models = {
        "Logistic Regression": jb.load("model_LogReg_PCA.pkl"),
        "KNN": jb.load("model_kNN_PCA.pkl"),
        "Random Forest": jb.load("model_RF_PCA.pkl")
    }
    return scaler, models

scaler, models = load_components()
# -------------------

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar buttons
with st.sidebar:
    st.title("Navigation")
    if st.button("Home"):
        st.session_state.page = "Home"
    if st.button("Dataset"):
        st.session_state.page = "Dataset"
    if st.button("Visualization"):
        st.session_state.page = "Visualization"
    if st.button("Prediction"):
        st.session_state.page = "Prediction"

# Current page
page = st.session_state.page

if page == "Home":
    st.title("Breast Cancer Prediction App")
    st.write("""
    This application predicts whether a breast tumor is:
    - Benign
    - Malignant

    using Machine Learning models:
    - Logistic Regression
    - K-Nearest Neighbors
    - Random Forest
    """)
    st.subheader("Dataset Shape")
    st.write(df.shape)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

elif page == "Dataset":
    st.title("Dataset")
    st.dataframe(df)
    st.subheader("Statistics")
    st.dataframe(df.describe())

elif page == "Visualization":
    st.title("Visualization")
    st.subheader("Diagnosis Distribution")
    diagnosis_count = df["diagnosis"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(["Benign", "Malignant"], diagnosis_count.values)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    # Hanya menghitung korelasi untuk kolom numerik agar tidak error
    numeric_df = df.select_dtypes(include=[np.number]) 
    corr = numeric_df.corr()
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    heatmap = ax2.imshow(corr)
    plt.colorbar(heatmap)
    st.pyplot(fig2)

elif page == "Prediction":
    st.title("Prediction")
    st.write("Input tumor measurements below.")

    model_name = st.segmented_control(
        "Choose Model",
        ["Logistic Regression", "KNN", "Random Forest"],
        default="Logistic Regression",
    )

    model = models[model_name]

    col1, col2 = st.columns(2)
    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, value=14.0)
        texture_mean = st.number_input("Texture Mean", min_value=0.0, value=19.0)
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=90.0)
        area_mean = st.number_input("Area Mean", min_value=0.0, value=600.0)
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1)

    with col2:
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0, value=0.1)
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, value=0.1)
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, value=0.05)
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.2)
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, value=0.06)

    if st.button("Predict"):
        # Menyesuaikan sisa fitur dummy agar total matriks input pas 34 fitur 
        # (10 input manual + 20 dummy + 4 feature engineering)
        remaining_features = [0] * 20

        # --- HITUNG FITUR TAMBAHAN (FEATURE ENGINEERING) ---
        area_perimeter_ratio = area_mean / perimeter_mean if perimeter_mean != 0 else 0
        radius_growth_ratio = 0 # Default karena radius_worst tidak diinput manual
        concavity_score = concavity_mean + concave_points_mean
        
        if radius_mean < 12:
            radius_category = 0
        elif 12 <= radius_mean < 18:
            radius_category = 1
        else:
            radius_category = 2

        # Susun array data sesuai urutan matriks training kalian
        input_data = np.array(
            [
                [
                    radius_mean,
                    texture_mean,
                    perimeter_mean,
                    area_mean,
                    smoothness_mean,
                    compactness_mean,
                    concavity_mean,
                    concave_points_mean,
                    symmetry_mean,
                    fractal_dimension_mean,
                    *remaining_features,       # Menutup kolom fitur 11 s.d 30
                    area_perimeter_ratio,      # Fitur 31
                    radius_growth_ratio,       # Fitur 32
                    concavity_score,           # Fitur 33
                    radius_category            # Fitur 34
                ]
            ]
        )

        # 1. Jalankan proses scaling menggunakan objek scaler bawaan
        input_scaled = scaler.transform(input_data)

        # 2. Jalankan prediksi (Pipeline akan otomatis memproses PCA 34 -> 11 fitur di balik layar)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("Malignant Tumor Detected")
        else:
            st.success("Benign Tumor Detected")

        st.subheader("Prediction Probability")
        st.write(f"Benign: {probability[0] * 100:.2f}%")
        st.write(f"Malignant: {probability[1] * 100:.2f}%")

        st.subheader("Selected Model")
        st.write(model_name)