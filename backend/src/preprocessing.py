def preprocess_data(df):
    df = df.drop_duplicates()

    df = df.drop(["Household_ID", "Duplicate_Flag", "Eligible"], axis=1)

    df["Aadhaar_Linked"] = df["Aadhaar_Linked"].map({
        "Yes": 1,
        "No": 0
    })

    return df