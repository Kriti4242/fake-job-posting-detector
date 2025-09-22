import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import seaborn as sns

# ======================================================
# 1. Load dataset
# ======================================================
df = pd.read_csv("fake_job_postings.csv")
print(f"Original dataset shape: {df.shape}")

# Subset for faster training
df = df.sample(5000, random_state=42)
print(f"Subset dataset shape: {df.shape}")

# ======================================================
# 2. Preprocessing
# ======================================================
df = df.fillna(" ")
df["text"] = (
    df["title"].astype(str)
    + " "
    + df["company_profile"].astype(str)
    + " "
    + df["description"].astype(str)
    + " "
    + df["requirements"].astype(str)
    + " "
    + df["benefits"].astype(str)
)

X = df["text"]
y = df["fraudulent"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X_vec = vectorizer.fit_transform(X)

# ======================================================
# 3. Train-test split
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=42
)

X_tr, X_va, y_tr, y_va = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Handle imbalance
spw = (y_train == 0).sum() / (y_train == 1).sum()

# ======================================================
# 4. Train XGBoost
# ======================================================
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=spw,
    random_state=42,
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_va, y_va)],
    verbose=False
)

# ======================================================
# 5. Evaluation
# ======================================================
preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
roc = roc_auc_score(y_test, proba)
pr, rc, f1, _ = precision_recall_fscore_support(
    y_test, preds, average="binary", zero_division=0
)
cm = confusion_matrix(y_test, preds)

print("\n📊 XGBoost Model Performance:")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print(f"Precision: {pr:.4f}")
print(f"Recall: {rc:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, preds))
print("Confusion Matrix:\n", cm)

# ======================================================
# 6. Baseline Logistic Regression
# ======================================================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]

lr_acc = accuracy_score(y_test, lr_preds)
lr_pr = precision_score(y_test, lr_preds, zero_division=0)
lr_rc = recall_score(y_test, lr_preds, zero_division=0)
lr_f1 = f1_score(y_test, lr_preds, zero_division=0)
lr_roc = roc_auc_score(y_test, lr_probs)

print("\n📊 Logistic Regression Baseline:")
print(f"Accuracy: {lr_acc:.4f}")
print(f"Precision: {lr_pr:.4f}")
print(f"Recall: {lr_rc:.4f}")
print(f"F1-score: {lr_f1:.4f}")
print(f"ROC-AUC: {lr_roc:.4f}")

# ======================================================
# 7. Visualizations
# ======================================================

# Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real","Fake"],
            yticklabels=["Real","Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"XGBoost ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, proba)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, lw=2, color="green")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - XGBoost")
plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.show()

# ======================================================
# 8. SHAP Explainability
# ======================================================
X_tr_dense = X_tr[:100].toarray()
X_test_dense = X_test[:30].toarray()

explainer = shap.Explainer(model, X_tr_dense, feature_names=vectorizer.get_feature_names_out())
shap_values = explainer(X_test_dense)

shap.summary_plot(shap_values, feature_names=vectorizer.get_feature_names_out(), show=False)
plt.savefig("shap_summary.png")
print("✅ SHAP summary plot saved as shap_summary.png")

# ======================================================
# 9. Save Model + Metrics
# ======================================================
bundle = {
    "model": model,
    "vectorizer": vectorizer,
    "metrics": {
        "accuracy": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "roc_auc": roc,
        "confusion_matrix": cm.tolist()
    },
    "X_test": X_test,
    "y_test": y_test
}

joblib.dump(bundle, "fake_job_detector.joblib")
print("✅ Model, metrics & test data saved as fake_job_detector.joblib")
