import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:\\Tekworks\\newproject\\backend\\data\\welfare_large_dataset.csv")

# Basic info
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Target distribution
sns.countplot(x="Fraud_Flag", data=df)
plt.title("Fraud Distribution")
plt.show()
# Income distribution
sns.histplot(df["Income"], kde=True)
plt.title("Income Distribution")
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.show()