import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="kameshwarink/Mlops-visit-with-us", filename="best_tourism_model_predict_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Prediction
st.title("Predicting Wellness Tourism Package Purchase App")
st.write("""
model that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them
Please enter the required data below to get a prediction.
""")

# User input
# age = st.selectbox("Age", ["H", "L", "M"])
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry", "Other"])
CityTier = st.selectbox("CityTier", ["1", "2", "3"])
DurationOfPitch = st.number_input("Duration", min_value=1, max_value=70, value=10)
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=6, value=2)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=2)
ProductPitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "King", "Super Deluxe"])
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=3, max_value=5, value=3)
MaritalStatus = st.selectbox("MaritalStatus", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1, max_value=50, value=2)
Passport = st.selectbox("Passport", ["0", "1"])
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=3)
OwnCar = st.number_input("OwnCar", min_value=0, max_value=1, value=0)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0)
Designation = st.selectbox("Designation", ["AVP", "VP", "Manager", "Executive", "Senior Manager"])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000, max_value=100000, value=5000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict Wellness Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Purchase" if prediction == 1 else "No Purchase"
    st.subheader("Prediction Result:")
    st.write(f"The predicted status for the customer is {prediction}")
    st.success(f"The model predicts: **{result}**")
