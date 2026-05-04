# fraud_detection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# ===============================
# 1. Create results folder
# ===============================
if not os.path.exists("results"):
    os.makedirs("results")

# ===============================
# 2. Load dataset
# ===============================
print("Loading dataset...")
data = pd.read_csv("creditcard.csv")

print("Dataset shape:", data.shape)
print("Fraud cases:", sum(data["Class"] == 1))
print("Normal cases:", sum(data["Class"] == 0))

# ===============================
# 3. Preprocessing
# ===============================
X = data.drop("Class", axis=1)   # Features
y = data["Class"]                # Target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 4. Train Isolation Forest
# ===============================
print("\nTraining Isolation Forest...")
model = IsolationForest(
    n_estimators=100,
    contamination=0.005,  # estimated fraud percentage in dataset
    random_state=42
)
model.fit(X_scaled)

# Predictions: -1 = anomaly, 1 = normal
y_pred = model.predict(X_scaled)
y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert: 1=fraud, 0=normal

# ===============================
# 5. Evaluation
# ===============================
print("\nModel Evaluation:")
print(classification_report(y, y_pred, target_names=["Normal", "Fraud"]))

cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"])
plt.title("Confusion Matrix - Isolation Forest")
plt.savefig("results/confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y, y_pred)
roc_auc = roc_auc_score(y, y_pred)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Isolation Forest")
plt.legend(loc="lower right")
plt.savefig("results/roc_curve.png")
plt.close()

print("\nResults saved in 'results/' folder.")