import pandas as pd
import streamlit as st
import joblib

# Load the trained model
model = joblib.load('Loan_Prediction_Model_DecisionTree.pkl')

# Title of the web app
st.title("Loan Prediction Model")

# Description of the app
st.write("Please fill out the following fields to get a loan prediction.")

# Input fields for user data
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=0)
credit_history = st.selectbox("Credit History", ["1.0", "0.0"])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Add a predict button
if st.button('Predict'):
    with st.spinner('Making prediction...'):
        try:
            # Convert input data to the right format
            input_data = pd.DataFrame({
                'Gender_T': [1 if gender == 'Male' else 0],
                'Married_T': [1 if married == 'Yes' else 0],
                'Dependents_T': [int(dependents.replace('+', ''))],
                'Education_T': [1 if education == 'Graduate' else 0],
                'Self_Employed_T': [1 if self_employed == 'Yes' else 0],
                'ApplicantIncome': [applicant_income],
                'CoapplicantIncome': [coapplicant_income],
                'LoanAmount': [loan_amount],
                'Loan_Amount_Term': [loan_amount_term],
                'Credit_History': [float(credit_history)],
                'Property_Area_T': [2 if property_area == 'Urban' else (1 if property_area == 'Semiurban' else 0)]
            })

            # Ensure input_data contains all the required features
            feature_columns = [ 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                               'Credit_History', 'Gender_T', 'Married_T','Dependents_T', 'Education_T','Self_Employed_T','Property_Area_T',]

            for col in feature_columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Reorder the columns to match the training feature order
            input_data = input_data[feature_columns]

            # Make prediction
            prediction = model.predict(input_data)

            # Display the result
            st.write(f"**Prediction:** {'Approved' if prediction[0] == 1 else 'Not Approved'}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
