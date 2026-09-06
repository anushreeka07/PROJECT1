import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("dataset/credit_card_fraud_10k.csv")

print("\n===== DATASET INFORMATION =====")
print("Dataset shape:", data.shape)

# Check missing values
print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

# Check duplicate records
print("\n===== DUPLICATE RECORDS =====")
print("Number of duplicate records:", data.duplicated().sum())

# Remove duplicate records
data = data.drop_duplicates().copy()

# Handle missing values
for column in data.columns:
    if data[column].isnull().any():
        if data[column].dtype == "object":
            data[column] = data[column].fillna(data[column].mode()[0])
        else:
            data[column] = data[column].fillna(data[column].median())

print("\nDataset after cleaning:", data.shape)

# -----------------------------
# 2. Convert Categorical Data
# -----------------------------
df = data.copy()
encoders = {}

for column in df.select_dtypes(include=["object"]).columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))
    encoders[column] = encoder

# -----------------------------
# 3. Separate Features and Target
# -----------------------------
target = "is_fraud"

if target not in df.columns:
    raise ValueError("Target column 'is_fraud' was not found in the dataset.")

X = df.drop(target, axis=1)
y = df[target]

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# -----------------------------
# 5. Train Random Forest Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 6. Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7. Model Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

print("\n===== CONFUSION MATRIX =====")
print(cm)

# -----------------------------
# 8. Save Predictions
# -----------------------------
results = X_test.copy()
results["actual_is_fraud"] = y_test.values
results["predicted_is_fraud"] = y_pred

results["actual_label"] = results["actual_is_fraud"].map(
    {0: "Legitimate", 1: "Potentially Fraudulent"}
)
results["predicted_label"] = results["predicted_is_fraud"].map(
    {0: "Legitimate", 1: "Potentially Fraudulent"}
)

results.to_csv("fraud_detection_predictions.csv", index=False)

# -----------------------------
# 9. Save Evaluation Report
# -----------------------------
report = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score": [accuracy, precision, recall, f1],
    "Percentage": [
        accuracy * 100,
        precision * 100,
        recall * 100,
        f1 * 100
    ]
})

report.to_csv("model_evaluation_report.csv", index=False)

print("\nFiles created:")
print("- fraud_detection_predictions.csv")
print("- model_evaluation_report.csv")
