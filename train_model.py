import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# ==================================
# LOAD DATASET
# ==================================

df = pd.read_csv("diabetes.csv")

print("Dataset shape:", df.shape)


# ==================================
# DATA CLEANING
# ==================================

# Replace impossible zeros with median
cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

for c in cols:
    df[c] = df[c].replace(0, df[c].median())


# ==================================
# FEATURES / TARGET
# ==================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ==================================
# TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ==================================
# MODEL PIPELINE
# ==================================

pipeline = Pipeline([

    ("scaler", StandardScaler()),

    ("model", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ))

])


# ==================================
# TRAIN MODEL
# ==================================

print("Training model...")

pipeline.fit(X_train, y_train)


# ==================================
# PREDICTION
# ==================================

pred = pipeline.predict(X_test)


# ==================================
# EVALUATION
# ==================================

accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test, pred))

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, pred))


# ==================================
# FEATURE IMPORTANCE
# ==================================

model = pipeline.named_steps["model"]

importance = model.feature_importances_

features = X.columns

os.makedirs("static", exist_ok=True)

plt.figure(figsize=(8,5))

plt.barh(features, importance)

plt.title("Feature Importance")

plt.xlabel("Importance")

plt.tight_layout()

plt.savefig("static/feature_importance.png")

print("\nFeature importance saved to static/feature_importance.png")


# ==================================
# SAVE MODEL
# ==================================

joblib.dump(pipeline, "diabetes_model.pkl")

print("\nModel saved as diabetes_model.pkl")