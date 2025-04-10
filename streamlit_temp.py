import streamlit as st
import pandas as pd
import joblib

# Load the entire pipeline (includes preprocessing and regression model)
model = joblib.load('salary_model.pkl')

# Streamlit app title
st.title("Salary Prediction App")

# Input fields for user data
age = st.number_input("Enter Age:", min_value=18, max_value=100, step=1)
balance = st.number_input("Enter Balance:", min_value=0.0, step=1000.0)
num_of_products = st.number_input("Number of Products:", min_value=1, max_value=10, step=1)
geography = st.selectbox("Select Geography:", ["France", "Germany", "Spain"])
gender = st.selectbox("Select Gender:", ["Male", "Female"])

# Predict salary when user clicks the button
if st.button("Predict"):
    # Create a DataFrame for input data
    input_data = pd.DataFrame({
        "Age": [age],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "Geography": [geography],
        "Gender": [gender]
    })

    # Use the pipeline for prediction
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Salary: {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")



