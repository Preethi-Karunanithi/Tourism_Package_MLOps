import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model


model_path = hf_hub_download(repo_id="PreethiKarunanithi/Preethi-first-project-space", filename="best_tourism_package_model_v1.joblib")

model = joblib.load(model_path)


# Streamlit UI for Tourism Package Prediction
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Tourism Package Prediction")
st.write("Predict whether a customer is likely to purchase a tourism package.")

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", min_value=18, max_value=100)

if age <= 18:
    age_group = "Child"
elif age <= 30:
    age_group = "Young Adult"
elif age <= 45:
    age_group = "Adult"
elif age <= 60:
    age_group = "Middle Aged"
else:
    age_group = "Senior"

# Outside the if-else block
pitch_satisfaction = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

children = st.number_input(
    "Number of Children Visiting",
    0,
    10,
    0
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

DurationOfPitch = st.number_input(
    "Duration of Pitch (Minutes)",
    min_value=1,
    max_value=60,
    value=15
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.slider(
    "Number of Persons Visiting",
    1, 10, 2
)

NumberOfFollowups = st.slider(
    "Number of Follow-ups",
    0, 10, 2
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

NumberOfTrips = st.slider(
    "Number of Trips",
    0, 20, 3
)

Passport = st.selectbox(
    "Passport",
    [0, 1]
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=1000,
    value=25000
)

# -----------------------------
# Create DataFrame
# -----------------------------
input_data = {
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": children,
    "MonthlyIncome": MonthlyIncome,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "ProductPitched": ProductPitched,
    "Designation": designation,
    "Age_Group": age_group
}

input_df = pd.DataFrame([input_data])

#st.write(input_df.columns.tolist())

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("✅ Customer is likely to purchase the tourism package.")
    else:
        st.error("❌ Customer is not likely to purchase the tourism package.")

    st.write(f"**Prediction Probability:** {probability:.2%}")

    st.subheader("Input Summary")
    st.dataframe(input_df)
