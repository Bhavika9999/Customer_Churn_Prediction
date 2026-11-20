
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score
)

# Load dataset
df = pd.read_csv("data/featured_telco_churn.csv")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# Identify columns
categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns

numerical_columns = X.select_dtypes(
    include=["number"]
).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define models and parameters
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"model__C": [0.1, 1, 10]}
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "model__max_depth": [5, 10, None],
            "model__min_samples_split": [2, 5]
        }
    )
}

best_score = 0
best_model = None
best_model_name = ""

# Train and tune models
for name, (model, params) in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipeline,
        params,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    predictions = grid.predict(X_test)
    probabilities = grid.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print("\nModel:", name)
    print("Best Parameters:", grid.best_params_)
    print("Cross Validation AUC:", grid.best_score_)
    print("Test ROC-AUC:", auc)
    print("Precision-Recall AUC:", pr_auc)
    print("Precision:", precision)
    print("Recall:", recall)

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_model_name = name

# Save best model
joblib.dump(best_model, "best_churn_model.pkl")

print("\nBest Model:", best_model_name)
print("Best model saved successfully!")
