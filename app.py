import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details and predict whether the customer is likely to churn.")

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

if st.button("Predict Churn"):

                input_data = pd.DataFrame({
                    "gender": [gender],
                    "SeniorCitizen": [senior],
                    "Partner": [partner],
                    "Dependents": [dependents],
                    "tenure": [tenure],
                    "PhoneService": [phone_service],
                    "MultipleLines": [multiple_lines],
                    "InternetService": [internet_service],
                    "OnlineSecurity": [online_security],
                    "OnlineBackup": [online_backup],
                    "DeviceProtection": [device_protection],
                    "TechSupport": [tech_support],
                    "StreamingTV": [streaming_tv],
                    "StreamingMovies": [streaming_movies],
                    "Contract": [contract],
                    "PaperlessBilling": [paperless],
                    "PaymentMethod": [payment],
                    "MonthlyCharges": [monthly_charges],
                    "TotalCharges": [total_charges]
                })

                input_encoded = pd.get_dummies(input_data)

                try:
                    feature_names = scaler.feature_names_in_

                    input_encoded = input_encoded.reindex(
                        columns=feature_names,
                        fill_value=0
                    )

                    input_scaled = scaler.transform(input_encoded)

                    probability = model.predict_proba(input_scaled)[0][1]
                    st.write("Raw Probability:", probability)
                    st.write("Encoded Input:")
                    st.dataframe(input_encoded)


                    # Custom threshold
                    prediction = 1 if probability >= 0.40 else 0

                    st.subheader("Prediction Result")

                    if prediction == 1:
                        st.error("⚠️ Customer is likely to churn")
                    else:
                        st.success("✅ Customer is likely to stay")

                    st.metric(
                        "Churn Probability",
                        f"{probability:.2%}"
                    )

                    if probability >= 0.70:
                        st.warning("🔴 High Risk Customer")
                    elif probability >= 0.40:
                        st.info("🟠 Medium Risk Customer")
                    else:
                        st.success("🟢 Low Risk Customer")

                except Exception as e:
                    st.error(f"Error: {e}")
