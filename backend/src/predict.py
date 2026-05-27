import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")


def predict(data):
    df = pd.DataFrame([data])

    cols_to_drop = ["Duplicate_Flag", "Eligible"]

    df = df.drop(columns=cols_to_drop, errors="ignore")

    prediction = int(model.predict(df)[0])
    confidence = float(model.predict_proba(df)[0][1])

    return prediction, confidence