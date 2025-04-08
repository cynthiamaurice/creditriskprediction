import pickle
import streamlit as st
from PIL import Image
import pandas as pd
import bz2

st.title("Credit Risk Prediction")
img = Image.open("C:\\Users\\personal\\Desktop\\data_prediction_model\\kuda logo.webp")

#load the model
def decompress_pickle(data_prediction_model):
    with bz2.BZ2File(data_prediction_model, 'rb') as f:
        data = pickle.load(f)
    return data
model = decompress_pickle("C:\\Users\\personal\\Desktop\\data_prediction_model.pbz2")
st.image(img, width=400)

# User input form
age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Annual Income", min_value=1)  # avoid divide by zero
emp_length = st.number_input("Employment Length (Years)", min_value=1.0)  # avoid divide by zero
loan_amount = st.number_input("Loan Amount", min_value=1)
int_rate = st.number_input("Interest Rate (%)", min_value=0.01)

# Derived ratios
loan_to_income_ratio = loan_amount / income
loan_to_emp_length_ratio = loan_amount / emp_length
int_rate_to_loan_amt_ratio = int_rate / loan_amount
# Derived features
loan_to_income_ratio = loan_amount / income if income > 0 else 0
loan_to_emp_length_ratio = loan_amount / emp_length if emp_length > 0 else 0
int_rate_to_loan_amt_ratio = int_rate / loan_amount if loan_amount > 0 else 0


# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    "person_age": [age],
    "person_income": [income],
    "person_emp_length": [emp_length],
    "loan_amnt": [loan_amount],
    "loan_int_rate": [int_rate],
    "loan_to_income_ratio": [loan_to_income_ratio],
    "loan_to_emp_length_ratio": [loan_to_emp_length_ratio],
    "int_rate_to_loan_amt_ratio": [int_rate_to_loan_amt_ratio]
})

# Make prediction
# Additional Threshold Logic
def check_income_loan_threshold(income, loan_amount):
    # Example: Reject loan if income is too low or loan amount is too high for that income
    if income < 50000:  # Example threshold for low income
        return False
    if loan_amount > income :  # Example threshold for loan amount > annual income
        return False
    return True

# Make prediction with custom threshold and additional rules
if st.button("Predict Loan Status"):
    if not check_income_loan_threshold(income, loan_amount):
        result = "❌ Rejected "
    else:
        # Use model to get the probability of rejection (class 1)
        proba = model.predict_proba(input_data)[0][1]  # Probability for class '1' (Rejected)
        threshold = 0.3  # Custom threshold for rejection probability
        predicted = 1 if proba > threshold else 0  # Use threshold for decision

        # Final decision based on both income/loan checks and model prediction
        if predicted == 0:
            result = "✅ Approved"
        else:
            result = "❌ Rejected "

    st.subheader(f"Loan Prediction Result: {result}")
