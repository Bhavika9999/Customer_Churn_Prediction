
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_telco_churn.csv")

# Feature 1: Tenure Group
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New", "Short-term", "Mid-term", "Long-term"]
)

# Feature 2: Has Family
df["HasFamily"] = (
    (df["Partner"] == "Yes") |
    (df["Dependents"] == "Yes")
).astype(int)

# Feature 3: Number of Services
service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["NumberOfServices"] = (
    df[service_columns] == "Yes"
).sum(axis=1)

# Display new features
print("New Features:")
print(df[["TenureGroup", "HasFamily", "NumberOfServices"]].head())

# Save featured dataset
df.to_csv("data/featured_telco_churn.csv", index=False)

print("\nFeature Engineering Completed!")
