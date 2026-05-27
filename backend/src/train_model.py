from sklearn.ensemble import RandomForestClassifier
import joblib

from data_ingestion import load_data
from preprocessing import preprocess_data
from feature_engineering import feature_engineer
from split_data import split_data


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,          # reduced from 200
        max_depth=6,               # reduced from 10
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/fraud_model.pkl")

    return model


if __name__ == "__main__":
    df = load_data("data/welfare_large_dataset.csv")

    df = preprocess_data(df)

    df, encoders = feature_engineer(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    print("Model trained successfully")