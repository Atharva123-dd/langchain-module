import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("titanic_pipeline.pkl")

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# User Inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=22
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=7.25
)

# Prediction button
if st.button("Predict"):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "Fare": [fare]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.success("✅ Passenger is predicted to SURVIVE")
    else:
        st.error("❌ Passenger is predicted to NOT SURVIVE")

    st.write("### Prediction Probability")

    st.write(f"Survive: **{probability[1]*100:.2f}%**")
    st.write(f"Not Survive: **{probability[0]*100:.2f}%**")