from sklearn.preprocessing import LabelEncoder
from data_ingestion import load_data
from preprocessing import preprocess_data

def feature_engineer(df):
    encoders = {}

    categorical_cols = ["State", "Occupation", "Age_Group"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


if __name__ == "__main__":
    df = load_data("data/welfare_large_dataset.csv")

    df = preprocess_data(df)

    df, encoders = feature_engineer(df)

    print(df.head())
    print(df.dtypes)