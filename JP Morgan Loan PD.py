# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 13:22:42 2025

@author: EBUNOLUWASIMI
"""

# loan_default_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# === STEP 1: Load data ===
df = pd.read_csv(r"C:\Users\EBUNOLUWASIMI\Dropbox\Study Materials\Python\JP Morgan Chase\Task 3 and 4_Loan_Data.csv")

# Inspect the first few rows to confirm structure
print(df.head())

# === STEP 2: Define features (X) and target (y) ===
# Adjust feature columns to match your dataset
feature_cols = [col for col in df.columns if col != "default" and col != "customer_id"]
X = df[feature_cols]
y = df["default"]

# === STEP 3: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 4: Train model ===
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# === STEP 5: Evaluate model ===
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of default

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))

# === STEP 6: Save model for future use ===
joblib.dump(model, "loan_default_model.pkl")
print("\nModel saved as loan_default_model.pkl")

# === STEP 7: Define Expected Loss function ===
def expected_loss(input_data, model, recovery_rate=0.1):
    """
    input_data: dictionary with borrower details matching feature columns
    model: trained classifier
    recovery_rate: % recovered after default (default 10%)
    """
    df_input = pd.DataFrame([input_data])
    pd_prob = model.predict_proba(df_input)[:, 1][0]
    loan_amount = input_data.get("loan_amount", 0)  # Adjust column name if needed
    el = pd_prob * (1 - recovery_rate) * loan_amount
    return {"PD": pd_prob, "Expected Loss": el}

# === STEP 8: Example usage ===
example_borrower = {
    "credit_lines_outstanding":4,
    "loan_amt_outstanding": 5713.43439,
    "total_debt_outstanding": 21401.27554,
    "income": 81627.55137,
    "years_employed":3,
    "fico_score": 533
    # Add any other features your dataset contains
}

model = joblib.load("loan_default_model.pkl")
result = expected_loss(example_borrower, model)
print("\nExample borrower prediction:")
print(result)
