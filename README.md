# 📊 Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to churn (leave the service) based on customer demographics, account information, and service usage.

## 🚀 Live Demo

[Streamlit App Link Here]

---

##  Project Overview

Customer churn is a major challenge for subscription-based businesses. This project uses Machine Learning to identify customers who are likely to leave, helping businesses take preventive actions.

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

---

##  Dataset

Dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges

Target Variable:

- Churn (Yes/No)

---

##  Data Preprocessing

- Handled missing values
- Converted categorical variables
- Feature encoding
- Feature scaling using StandardScaler
- Train-test split

---

##  Models Evaluated

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 79.91% |
| Random Forest | 78.78% |
| Decision Tree | 72.96% |

Selected Model:

**Logistic Regression**

---

##  Model Performance

- Accuracy: 79.91%
- Precision (Churn): 0.63
- Recall (Churn): 0.49
- F1-Score (Churn): 0.55

---

##  Features

- Single Customer Churn Prediction
- Interactive Streamlit Dashboard
- Churn Probability Score
- Risk Level Classification
- User-Friendly Interface

---

##  Screenshots

### Home Page

(Add screenshot here)

### Prediction Result

(Add screenshot here)

---

##  Run Locally

Clone the repository:

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

##  Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── train_model.py
├── Customer_Churn_Analysis.ipynb
├── churn_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

##  Author

Pami Gupta