import pickle
import streamlit as st
from PIL import Image
import pandas as pd
import bz2



st.title("Credit Risk Prediction")
img = Image.open(r"logo.webp")

#load the model
def decompress_pickle(data_prediction_model):
    with bz2.BZ2File(data_prediction_model, 'rb') as f:
        data = pickle.load(f)
    return data
model = decompress_pickle(r"data_prediction_model.pbz2")
st.image(img, width=400)

# User input form
name = st.text_input("what is your full name?")
age = st.number_input("Age", min_value=20, max_value=60)
income = st.number_input("Annual Income", min_value=1)  # avoid divide by zero
emp_length = st.number_input("Employment Length (Years)", min_value=1)  # avoid divide by zero
loan_amount = st.number_input("Loan Amount", min_value=1)
loan_duration = st.number_input("Loan Duration (Months)", min_value=1,max_value=18)



# Fixed interest rate
int_rate = 0.35  # 35%
monthly_interest_rate = int_rate / 12
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
# Threshold check
def check_income_loan_threshold(income_val, user_loan_amt):
    return income_val >= 50000 and user_loan_amt < income_val

# Reducing balance schedule
def reducing_balance_schedule(principal, annual_rate, months):
    monthly_rate = annual_rate / 12 # noqa: 57
    schedule = []
    outstanding = principal

    monthly_payment = (principal * monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)

    for month in range(1, months + 1): # noqa: 63
        interest_payment = outstanding * monthly_rate # noqa: 64
        principal_payment = monthly_payment - interest_payment # noqa: 65
        outstanding -= principal_payment

        schedule.append({
            "Month": month,
            "Payment": round(monthly_payment, 2),
            "Interest": round(interest_payment, 2),
            "Principal": round(principal_payment, 2),
            "Balance": round(outstanding if outstanding > 0 else 0, 2)
        })

        if outstanding <= 0:
            break

    return pd.DataFrame(schedule)

result = None  # Initialize result so it's defined outside

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

# Show repayment schedule only if approved
if result == "✅ Approved":
    duration_months = loan_duration  # Make sure `loan_duration` is already captured from user input
    monthly_rate = int_rate / 12  # Monthly interest rate
    balance = qualified_loan_amount
    monthly_schedule = []

    for month in range(1, duration_months + 1):
        interest_payment = balance * monthly_rate
        principal_payment = (qualified_loan_amount / duration_months)
        total_payment = interest_payment + principal_payment
        balance -= principal_payment

        monthly_schedule.append({
            "Month": month,
            "Interest": round(interest_payment, 2),
            "Principal": round(principal_payment, 2),
            "Total Payment": round(total_payment, 2),
            "Remaining Balance": round(balance if balance > 0 else 0, 2)
        })

    schedule_df = pd.DataFrame(monthly_schedule)
    st.write("📊 **Repayment Schedule (Reducing Balance)**")
    st.dataframe(schedule_df)

    # Summary Totals
    total_interest = sum(item["Interest"] for item in monthly_schedule)
    total_principal = sum(item["Principal"] for item in monthly_schedule)
    total_repayment = sum(item["Total Payment"] for item in monthly_schedule)

    st.markdown("### 📈 Loan Summary")
    st.markdown(f"**Total Principal:** ₦{total_principal:,.2f}")
    st.markdown(f"**Total Interest:** ₦{total_interest:,.2f}")
    st.markdown(f"**Total Repayment:** ₦{total_repayment:,.2f}")
