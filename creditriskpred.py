import pickle
import streamlit as st
from PIL import Image
import pandas as pd
import bz2

from pyarrow import duration

st.title("Credit Risk Prediction")
img = Image.open(r"C:\Users\personal\Desktop\data_prediction_model\logo.webp")

#load the model
def decompress_pickle(data_prediction_model):
    with bz2.BZ2File(data_prediction_model, 'rb') as f:
        data = pickle.load(f)
    return data
model = decompress_pickle(r"C:\Users\personal\Desktop\data_prediction_model.pbz2")
st.image(img, width=400)

# User input form
name = st.text_input("what is your full name?")
age = st.number_input("Age", min_value=20, max_value=60)
income = st.number_input("Annual Income", min_value=1)  # avoid divide by zero
emp_length = st.number_input("Employment Length (Years)", min_value=1)  # avoid divide by zero
loan_amount = st.number_input("Loan Amount", min_value=1)


# Fixed interest rate
int_rate = 0.35  # 35%

# Calculate how much applicant qualifies for (e.g. 30% of income)
qualified_loan_amount = income * 0.3

# Derived ratios
loan_to_income_ratio = qualified_loan_amount / income if income > 0 else 0
loan_to_emp_length_ratio = qualified_loan_amount / emp_length if emp_length > 0 else 0
int_rate_to_loan_amt_ratio = int_rate / qualified_loan_amount if qualified_loan_amount > 0 else 0

# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    "person_age": [age],
    "person_income": [income],
    "person_emp_length": [emp_length],
    "loan_amnt": [loan_amount], # noqa: 40
    "loan_int_rate": [int_rate],
    "loan_to_income_ratio": [loan_to_income_ratio],
    "loan_to_emp_length_ratio": [loan_to_emp_length_ratio],
    "int_rate_to_loan_amt_ratio": [int_rate_to_loan_amt_ratio]
})
# Custom rule: Check income and loan thresholds
def check_income_loan_threshold(income, loan_amount): # noqa: 47
    if income < 50000:
        return False
    if loan_amount >= income:
        return False
    return True

# Make prediction with custom threshold and additional rules
if st.button("Predict Loan Status"):
    if not check_income_loan_threshold(income, loan_amount):
        # If the threshold condition fails, reject the loan without calling the model
        result = "❌ Rejected"
    else:
        # Otherwise, make a prediction using the model
        proba = model.predict_proba(input_data)[0][1]  # Probability for class '1' (Rejected)
        threshold = 0.3  # Custom threshold for rejection probability
        predicted = 1 if proba > threshold else 0  # Use threshold for decision

        # Final decision based on both income/loan checks and model prediction
        if predicted == 0:
            result = "✅ Approved"
        else:
            result = "❌ Rejected"
    st.subheader(f"Hi💜{name}!")
    st.subheader(f"Loan Prediction Result: {result}")

# Show how much the applicant qualifies for
st.write(f"💰 Based on your income, you're qualified to access up to: ₦{qualified_loan_amount:,.2f}")