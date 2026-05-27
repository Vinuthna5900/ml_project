from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from data_ingestion import load_data
from preprocessing import preprocess_data
from feature_engineering import feature_engineer
from split_data import split_data
from train_model import train_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    df = load_data("data/welfare_large_dataset.csv")

    df = preprocess_data(df)

    df, encoders = feature_engineer(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)
