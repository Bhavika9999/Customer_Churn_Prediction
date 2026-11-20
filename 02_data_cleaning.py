
import pandas as pd

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Check duplicate records
print("Duplicate Records:")
print(df.duplicated().sum())

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Check missing values after conversion
print("\nMissing Values:")
print(df.isnull().sum())

# Remove customerID
df = df.drop(columns=["customerID"])

# Check Churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# Dataset information
print("\nFinal Dataset Shape:")
print(df.shape)

print("\nData Cleaning Completed!")

df.to_csv("data/cleaned_telco_churn.csv", index=False)

print("Cleaned dataset saved successfully!")
