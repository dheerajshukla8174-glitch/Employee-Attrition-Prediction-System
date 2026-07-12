import streamlit as st
import pandas as pd
import joblib
# Load model and scaler
model = joblib.load("employee_attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
st.title("Employee Attrition Prediction")
monthly_income = st.number_input("Monthly Income", min_value=1000, value=5000)
age = st.number_input("Age", min_value=18, max_value=60, value=30)
overtime = st.selectbox("OverTime", ["No", "Yes"])
overtime = 1 if overtime == "Yes" else 0
total_working_years = st.number_input("Total Working Years", min_value=0, value=5)
distance_from_home = st.number_input("Distance From Home", min_value=1, value=10)
years_at_company = st.number_input("Years At Company", min_value=0, value=5)
if st.button("Predict"):
    data = pd.DataFrame({
        "MonthlyIncome": [monthly_income],
        "Age": [age],
        "OverTime": [overtime],
        "TotalWorkingYears": [total_working_years],
        "DistanceFromHome": [distance_from_home],
        "YearsAtCompany": [years_at_company]
    })
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    if prediction[0] == 1:
        st.error("Employee is likely to leave the company.")
    else:
        st.success("Employee is likely to stay in the company.")
        st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:15px;">
        Developed by <b>Dheeraj Shukla 
    </div>
    """,
    unsafe_allow_html=True
)