
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("data/featured_telco_churn.csv")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# Recreate the same test data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load best trained model
model = joblib.load("best_churn_model.pkl")

# Predict churn probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("BUSINESS IMPACT ANALYSIS")
print("=" * 65)

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print("Correctly Identified Churn:", tp)
    print("Missed Churn Customers:", fn)
    print("False Churn Alerts:", fp)
    print("Correctly Identified Non-Churn:", tn)
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 Score:", round(f1, 3))

print("\nBusiness Impact Analysis Completed!")
