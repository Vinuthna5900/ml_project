from sklearn.model_selection import train_test_split
from data_ingestion import load_data
from preprocessing import preprocess_data
from feature_engineering import feature_engineer


def split_data(df):
    X = df.drop("Fraud_Flag", axis=1)
    y = df["Fraud_Flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data("data/welfare_large_dataset.csv")

    df = preprocess_data(df)

    df, encoders = feature_engineer(df)

    X_train, X_test, y_train, y_test = split_data(df)

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)