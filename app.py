"""
Loan Approval Prediction App
-----------------------------
Deploys the trained AdaBoostClassifier (with a DecisionTree base
estimator) from the loan approval project as an interactive
Streamlit web app.

Required files (must sit in the same folder as this script):
    model.pkl   -> trained AdaBoostClassifier
    scaler.pkl  -> the MinMaxScaler fitted on the training features

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"

# Feature order the model was trained on (confirmed from the pickled
# model's feature_names_in_ attribute) — DO NOT reorder these.
FEATURE_ORDER = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
]

# Label-encoding maps, taken from the encoding scheme used in the
# original notebook (sklearn's LabelEncoder assigns codes in
# alphabetical order of the category names).
GENDER_MAP = {"Female": 0, "Male": 1}
EDUCATION_MAP = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
HOME_OWNERSHIP_MAP = {"Mortgage": 0, "Other": 1, "Own": 2, "Rent": 3}
LOAN_INTENT_MAP = {
    "Debt Consolidation": 0,
    "Education": 1,
    "Home Improvement": 2,
    "Medical": 3,
    "Personal": 4,
    "Venture": 5,
}
PREVIOUS_LOAN_MAP = {"No": 0, "Yes": 1}


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("💰 Loan Approval Predictor")
st.write(
    "Enter applicant details below to predict whether a loan is likely "
    "to be **Approved** or **Rejected**, based on a trained AdaBoost "
    "classification model."
)

if not MODEL_PATH.exists():
    st.error("`model.pkl` was not found in the app folder. Please add it and reload.")
    st.stop()

if not SCALER_PATH.exists():
    st.error(
        "`scaler.pkl` was not found in the app folder. The model was trained on "
        "MinMax-scaled features, so predictions cannot be made without the saved "
        "scaler. In your training notebook, run:\n\n"
        "`pickle.dump(mx, open('scaler.pkl', 'wb'))`\n\n"
        "then place the resulting file next to this app."
    )
    st.stop()

model = load_model()
scaler = load_scaler()

st.divider()
st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
    education = st.selectbox("Education Level", list(EDUCATION_MAP.keys()))
    person_income = st.number_input(
        "Annual Income ($)", min_value=0, value=50000, step=1000
    )
    employee_experience = st.number_input(
        "Employment Experience (years)", min_value=0, max_value=50, value=5, step=1
    )
    home_ownership = st.selectbox("Home Ownership", list(HOME_OWNERSHIP_MAP.keys()))
    previous_loan = st.selectbox("Previous Loan Defaults on File", list(PREVIOUS_LOAN_MAP.keys()))

with col2:
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=10000, step=500)
    loan_intent = st.selectbox("Loan Intent", list(LOAN_INTENT_MAP.keys()))
    loan_interest_rate = st.number_input(
        "Loan Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1
    )
    loan_percentage = st.slider(
        "Loan Percentage of Income (0 - 1)", min_value=0.0, max_value=1.0, value=0.2, step=0.01
    )
    credit_history = st.number_input(
        "Credit History Length (years)", min_value=0, max_value=50, value=5, step=1
    )
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=900, value=650, step=1
    )

st.divider()

if st.button("Predict Loan Status", type="primary", use_container_width=True):
    # Build feature vector in the exact order the model expects
    input_dict = {
        "person_age": age,
        "person_gender": GENDER_MAP[gender],
        "person_education": EDUCATION_MAP[education],
        "person_income": person_income,
        "person_emp_exp": employee_experience,
        "person_home_ownership": HOME_OWNERSHIP_MAP[home_ownership],
        "loan_amnt": loan_amount,
        "loan_intent": LOAN_INTENT_MAP[loan_intent],
        "loan_int_rate": loan_interest_rate,
        "loan_percent_income": loan_percentage,
        "cb_person_cred_hist_length": credit_history,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": PREVIOUS_LOAN_MAP[previous_loan],
    }

    input_df = pd.DataFrame([input_dict], columns=FEATURE_ORDER)

    try:
        # IMPORTANT: use transform(), never fit_transform(), at inference time
        scaled_input = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Error while scaling input: {e}")
        st.stop()

    prediction = model.predict(scaled_input)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled_input)[0]

    st.subheader("Result")
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    if proba is not None:
        st.write(f"Confidence — Approved: **{proba[1]*100:.1f}%**, Rejected: **{proba[0]*100:.1f}%**")

    with st.expander("View input sent to the model"):
        st.dataframe(input_df)

st.divider()
st.caption(
    "Model: AdaBoostClassifier (DecisionTree base estimator) · "
    "Preprocessing: MinMaxScaler · Built with Streamlit"
)
