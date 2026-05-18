import streamlit as st
import joblib as jb
import pandas as pd

# model_kNN = jb.load("kNN.pkl")
# model_RF = jb.load("RF.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")

# dataset = test_dataloader()

# df = pd.DataFrame(
#     dataset.data,
#     columns=dataset.feature_names
# )

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

    # st.write(df.shape)

    st.subheader("Dataset Preview")

    # st.dataframe(df.head())


elif page == "Dataset":

    st.title("Dataset")

    st.dataframe(df)

    st.subheader("Statistics")

    st.dataframe(df.describe())

elif page == "Visualization":

    import matplotlib.pyplot as plt

    st.title("Visualization")

    st.subheader("Diagnosis Distribution")

    diagnosis_count = df["diagnosis"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(["Benign", "Malignant"], diagnosis_count.values)

    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    corr = df.corr()

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

    # model = models[model_name]

   
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

        concave_points_mean = st.number_input(
            "Concave Points Mean", min_value=0.0, value=0.05
        )

        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.2)

        fractal_dimension_mean = st.number_input(
            "Fractal Dimension Mean", min_value=0.0, value=0.06
        )

  

    if st.button("Predict"):

     
        remaining_features = [0] * 20

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
                    *remaining_features, #add more later(im lazy)
                ]
            ]
        )

        # Scaling
        input_scaled = scaler.transform(input_data)

        # Prediction
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
