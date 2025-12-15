import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64


# --------------------------------------------------
# Download CSV helper
# --------------------------------------------------
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" '
        f'download="predictions.csv">Download Predictions CSV</a>'
    )
    return href


# --------------------------------------------------
# Title & Tabs
# --------------------------------------------------
st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])


# ==================================================
# TAB 1 : SINGLE PREDICTION
# ==================================================
with tab1:

    # -----------------------------
    # User Inputs
    # -----------------------------
    age = st.number_input("Age (years)", min_value=0, max_value=150)

    sex = st.selectbox("Sex", ["Male", "Female"])

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=0,
        max_value=300
    )

    cholesterol = st.number_input(
        "Serum Cholesterol (mm/dl)",
        min_value=0
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        ["<= 120 mg/dl", "> 120 mg/dl"]
    )

    resting_ecg = st.selectbox(
        "Resting ECG Results",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )

    max_hr = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=202
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Yes", "No"]
    )

    oldpeak = st.number_input(
        "Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=10.0
    )

    st_slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )

    # -----------------------------
    # Encoding
    # -----------------------------
    sex = 0 if sex == "Male" else 1

    chest_pain = [
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic",
        "Typical Angina"
    ].index(chest_pain)

    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0

    resting_ecg = [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ].index(resting_ecg)

    exercise_angina = 1 if exercise_angina == "Yes" else 0

    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # -----------------------------
    # Input DataFrame
    # -----------------------------
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    # -----------------------------
    # Models
    # -----------------------------
    algonames = [
        'Decision Trees',
        'Logistic Regression',
        'Support Vector Machine'
    ]

    modelnames = [
        'DecisionR.pkl',
        'LogisticR.pkl',
        'SVMR.pkl'
    ]

    predictions = []

    def predict_heart_disease(data):
        predictions.clear()
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    # -----------------------------
    # Submit
    # -----------------------------
    if st.button("Submit"):
        st.subheader("Results")
        st.markdown("-------------------------")

        result = predict_heart_disease(input_data)

        for i in range(len(predictions)):
            st.subheader(algonames[i])

            if result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")

            st.markdown("-------------------------")


# ==================================================
# TAB 2 : BULK CSV PREDICTION
# ==================================================
with tab2:

    st.title("Upload CSV File")

    st.subheader("Instructions to note before uploading the file:")
    st.info(
        """
        1. No NaN values allowed.
        2. Total 11 features in this order:
           ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'ST_Slope')
        3. Check the spellings of the feature names.
        4. Feature value conventions are same as training data.
        """
    )

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:

        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open("LogisticR.pkl", "rb"))

        expected_columns = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'ST_Slope'
        ]

        if set(expected_columns).issubset(input_data.columns):

            input_data["Prediction_LR"] = ""

            for i in range(len(input_data)):
                arr = input_data.iloc[i][expected_columns].values.reshape(1, -1)
                input_data.loc[i, "Prediction_LR"] = model.predict(arr)[0]

            input_data.to_csv("PredictedHeartLR.csv", index=False)

            st.subheader("Predictions")
            st.write(input_data)

            st.markdown(
                get_binary_file_downloader_html(input_data),
                unsafe_allow_html=True
            )

        else:
            st.warning("Please make sure the uploaded CSV has correct columns.")

    else:
        st.info("Upload a CSV file to get predictions.")
        
        
        
# ==================================================
# TAB 3 : MODEL INFORMATION
# ==================================================
with tab3:

    st.title("Model Performance Comparison")

    st.write(
        """
        This section shows the accuracy comparison of different
        machine learning models used for heart disease prediction.
        """
    )

    import plotly.express as px

    # Model accuracy data (use YOUR actual scores)
    data = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Support Vector Machine': 83.91
    }

    models = list(data.keys())
    accuracies = list(data.values())

    df = pd.DataFrame({
        'Models': models,
        'Accuracy (%)': accuracies
    })

    fig = px.bar(
        df,
        x='Models',
        y='Accuracy (%)',
        color='Models',
        text='Accuracy (%)',
        title='Accuracy Comparison of Models'
    )

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)


