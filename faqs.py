import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest classifier model
best_model_rf = joblib.load("best_model.pkl")

def calculate_bmi(weight, height):
    try:
        bmi = weight / ((height / 100) ** 2)
        return bmi
    except ZeroDivisionError:
        # Handle division by zero error
        return "Please provide valid input in all parameters"

def calculate_waist_hip_ratio(waist, hip):
    try:
        whratio = waist/ hip
        return whratio
    except ZeroDivisionError:
        # Handle division by zero error
        return "Please provide valid input in all parameters"

def app():
    st.title("PCOS Detection System")
    st.write("Please enter the following information:")

    # Collect input data from the user
    cycle_ri = st.radio("Cycle Regular/Irregular", options=["Regular", "Irregular"])
    cycle_length = st.number_input("Cycle Length (days)")
    weight_gain = st.radio("Weight Gain (Y/N)", options=["Yes", "No"])
    hair_growth = st.radio("Hair Growth (Y/N)", options=["Yes", "No"])
    skin_darkening = st.radio("Skin Darkening (Y/N)", options=["Yes", "No"])
    hair_loss = st.radio("Hair Loss (Y/N)", options=["Yes", "No"])
    pimples = st.radio("Pimples (Y/N)", options=["Yes", "No"])
    age = st.number_input("Age (yrs)")
    weight = st.number_input("Weight (Kg)")
    height = st.number_input("Height (Cm)")
    waist = st.number_input("Waist (inch)")
    hip = st.number_input("Hip (inch)")

    # Convert categorical inputs to numerical values
    cycle_ri = 1 if cycle_ri == "Regular" else 0
    weight_gain = 1 if weight_gain == "Yes" else 0
    hair_growth = 1 if hair_growth == "Yes" else 0
    skin_darkening = 1 if skin_darkening == "Yes" else 0
    hair_loss = 1 if hair_loss == "Yes" else 0
    pimples = 1 if pimples == "Yes" else 0

    # Calculate BMI and waist-to-hip ratio
    bmi = calculate_bmi(weight, height)
    waist_hip_ratio = calculate_waist_hip_ratio(waist, hip)

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Cycle(R/I)': [cycle_ri],
        'Cycle length(days)': [cycle_length],
        'Weight gain(Y/N)': [weight_gain],
        'hair growth(Y/N)': [hair_growth],
        'Skin darkening (Y/N)': [skin_darkening],
        'Hair loss(Y/N)': [hair_loss],
        'Pimples(Y/N)': [pimples],
        'Age (yrs)': [age],
        'Weight (Kg)': [weight],
        'Height(Cm)': [height],
        'BMI': [bmi],
        'Waist:Hip Ratio': [waist_hip_ratio]
    })

    if st.button("Predict"):
        # Predict PCOS using the trained model
        prediction = best_model_rf.predict(input_data)

        # Display the prediction result
        if prediction[0] == 1:
            st.warning("You are at high risk of having PCOS. Please consult a healthcare professional.")
        else:
            st.success("You are not at high risk of having PCOS. However, it's recommended to consult a healthcare professional for further evaluation.")

if __name__ == "__main__":
    app()
