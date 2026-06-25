from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import joblib 

df = pd.read_csv("Customer-Churn dataset.csv");

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, pred_lr))

# Save the model and scaler
joblib.dump(lr, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")