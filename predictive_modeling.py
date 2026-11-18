# PREDICTIVE MODELING USING MACHINE LEARNING
# Thiranex Internship Task 2

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Create output folder
os.makedirs("outputs", exist_ok=True)

print("=" * 50)
print("PREDICTIVE MODELING USING MACHINE LEARNING")
print("=" * 50)

# 1. Load Dataset

data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

print("\nDataset Loaded Successfully!")

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

# 2. Data Visualization

plt.figure(figsize=(7, 5))

sns.countplot(x="target", data=df)

plt.title("Target Distribution")

plt.xlabel("Target Class")

plt.ylabel("Count")

plt.savefig("outputs/target_distribution.png")

plt.show()

# 3. Data Preprocessing

X = df.drop("target", axis=1)

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)

print("Testing Data:", X_test.shape)

# 4. Decision Tree Model

dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

dt_accuracy = accuracy_score(
    y_test,
    dt_predictions
)

print("\nDecision Tree Accuracy:",
      round(dt_accuracy * 100, 2), "%")

# 5. Random Forest Model

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\nRandom Forest Accuracy:",
      round(rf_accuracy * 100, 2), "%")

# 6. Accuracy Comparison

models = ["Decision Tree", "Random Forest"]

accuracies = [
    dt_accuracy * 100,
    rf_accuracy * 100
]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")

plt.xlabel("Machine Learning Models")

plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

plt.savefig("outputs/accuracy_comparison.png")

plt.show()

# 7. Confusion Matrix

cm = confusion_matrix(y_test, rf_predictions)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted Values")

plt.ylabel("Actual Values")

plt.savefig("outputs/confusion_matrix.png")

plt.show()

# 8. ROC Curve

probabilities = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    probabilities
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest AUC = {roc_auc:.2f}"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("outputs/roc_curve.png")

plt.show()

# 9. Classification Report

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=data.target_names
    )
)

# 10. New Prediction

sample = X_test.iloc[[0]]

prediction = rf_model.predict(sample)

print("\nNew Prediction:")

if prediction[0] == 0:
    print("Predicted Outcome: Malignant")
else:
    print("Predicted Outcome: Benign")

# 11. Final Results

print("\n" + "=" * 50)

print("PROJECT COMPLETED SUCCESSFULLY!")

print("=" * 50)

print("Decision Tree Accuracy:",
      round(dt_accuracy * 100, 2), "%")

print("Random Forest Accuracy:",
      round(rf_accuracy * 100, 2), "%")

print("ROC AUC Score:",
      round(roc_auc, 2))

print("\nAll graphs saved inside outputs folder.")