# train_model_final.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
import os

# ============================
# 1. Load Dataset
# ============================
CSV_PATH = r"C:\Users\mhdaz\Downloads\ML PROJECT MCA\ML PROJECT MCA\HeartDisease\heart_disease.csv"  # update if needed

df = pd.read_csv(CSV_PATH)
print("Loaded dataset:", df.shape)

# ============================
# 2. Select Relevant Columns
# ============================
feature_order = [
    "Age", "Blood Pressure", "Cholesterol Level", "Sleep Hours",
    "Exercise Habits", "Smoking", "High Blood Pressure",
    "Low HDL Cholesterol", "Alcohol Consumption"
]
target_col = "Heart Disease Status"

missing = [c for c in feature_order + [target_col] if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns in dataset: {missing}")

df = df[feature_order + [target_col]]

# ============================
# 3. Handle Missing Values
# ============================
imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# ============================
# 4. Encode Categorical Features
# ============================
exercise_map = {
    "None": 0, "none": 0, "No": 0, "no": 0,
    "Moderate": 1, "moderate": 1, "Occasional": 1, "occasional": 1,
    "Regular": 2, "regular": 2, "Often": 2, "often": 2
}

binary_map = {
    "Yes": 1, "yes": 1, "Y": 1, "y": 1, 1: 1,
    "No": 0, "no": 0, "N": 0, "n": 0, 0: 0
}

def map_exercise(val):
    s = str(val).strip()
    return exercise_map.get(s, exercise_map.get(s.capitalize(), 0))

def map_binary(val):
    s = str(val).strip()
    return binary_map.get(s, binary_map.get(s.capitalize(), 0))

df["Exercise Habits"] = df["Exercise Habits"].apply(map_exercise)
df["Smoking"] = df["Smoking"].apply(map_binary)
df["High Blood Pressure"] = df["High Blood Pressure"].apply(map_binary)
df["Low HDL Cholesterol"] = df["Low HDL Cholesterol"].apply(map_binary)
df["Alcohol Consumption"] = df["Alcohol Consumption"].apply(map_binary)

for c in ["Age", "Blood Pressure", "Cholesterol Level", "Sleep Hours"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# ============================
# 5. Encode Target Column
# ============================
df[target_col] = df[target_col].apply(
    lambda x: 1 if str(x).strip().lower() in ("yes", "1", "y", "true", "t") else 0
)

# ============================
# 6. Train-Test Split
# ============================
X = df[feature_order]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 7. Train Random Forest Model
# ============================
model = RandomForestClassifier(
    n_estimators=200, max_depth=12, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# ============================
# 8. Evaluate Model
# ============================
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ============================
# 9. Confusion Matrix
# ============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ============================
# 10. ROC Curve
# ============================
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# ============================
# 11. Feature Importance
# ============================
importances = pd.Series(model.feature_importances_, index=feature_order).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ============================
# 12. Save Model & Metadata
# ============================
joblib.dump(model, "random_forest_model.pkl")
meta = {
    "feature_order": feature_order,
    "exercise_map": exercise_map,
    "binary_map": binary_map
}
joblib.dump(meta, "model_metadata.pkl")
print("\n✅ Model and metadata saved successfully!")
