import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. TEXT CLEANING
# ============================================================

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 2. LOAD DATASET AND REPLICATE EXACT SPLIT
# ============================================================

print("Loading dataset...")
df = pd.read_csv("dataset/DataSet.csv")

text_columns = ["title", "company_profile", "description", "requirements", "benefits"]
df["combined_text"] = df[text_columns].fillna("").agg(" ".join, axis=1)
df["clean_text"] = df["combined_text"].apply(clean_text)
df["fraudulent"] = df["fraudulent"].map({"f": 0, "t": 1})

# Exact stratified splits
X_train_full, X_test_full, y_train_full, y_test = train_test_split(
    df,
    df["fraudulent"],
    test_size=0.2,
    random_state=42,
    stratify=df["fraudulent"]
)

X_train_part, X_val_part, y_train_part, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print(f"Test set shape: {X_test_full.shape}")


# ============================================================
# 3. LOAD ARTIFACTS
# ============================================================

model = joblib.load("models/adaboost_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
threshold = joblib.load("models/threshold.pkl")

X_test_tfidf = tfidf.transform(X_test_full["clean_text"])
test_scores = model.decision_function(X_test_tfidf)
y_test_pred = (test_scores >= threshold).astype(int)

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# ============================================================
# 4. PLOT AND SAVE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(7, 5.5), dpi=300)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine (0)", "Fraudulent (1)"],
    yticklabels=["Genuine (0)", "Fraudulent (1)"],
    cbar=True,
    annot_kws={"size": 14, "weight": "bold"}
)
plt.title(f"AdaBoost Test Confusion Matrix (Threshold = {threshold})", fontsize=14, pad=12, fontweight="bold")
plt.xlabel("Predicted Class", fontsize=12, labelpad=8)
plt.ylabel("Actual True Class", fontsize=12, labelpad=8)
plt.tight_layout()
os.makedirs("models", exist_ok=True)
plt.savefig("confusion_matrix.png")
plt.close()
print("Saved confusion_matrix.png successfully!")


# ============================================================
# 5. ERROR ANALYSIS: FALSE POSITIVES & FALSE NEGATIVES
# ============================================================

test_results = X_test_full.copy()
test_results["actual"] = y_test
test_results["predicted"] = y_test_pred
test_results["decision_score"] = test_scores

# False Positives: Actual Genuine (0), Predicted Fraudulent (1)
fps = test_results[(test_results["actual"] == 0) & (test_results["predicted"] == 1)]

# False Negatives: Actual Fraudulent (1), Predicted Genuine (0)
fns = test_results[(test_results["actual"] == 1) & (test_results["predicted"] == 0)]

print("\n" + "=" * 65)
print(f"FALSE POSITIVES ANALYSIS (Count = {len(fps)})")
print("Genuine jobs incorrectly flagged as Fraudulent")
print("=" * 65)
for idx, (_, row) in enumerate(fps.iterrows(), 1):
    print(f"\n[FP #{idx}] Title: {row['title']}")
    print(f"Company Profile (len={len(str(row['company_profile']))}): {str(row['company_profile'])[:120]}...")
    print(f"Score: {row['decision_score']:.4f} (Threshold: {threshold})")
    print(f"Has logo: {row['has_company_logo']}, Telecommuting: {row['telecommuting']}")

print("\n" + "=" * 65)
print(f"FALSE NEGATIVES ANALYSIS (Count = {len(fns)})")
print("Fraudulent jobs missed by model (predicted as Genuine)")
print("=" * 65)
for idx, (_, row) in enumerate(fns.head(8).iterrows(), 1):
    print(f"\n[FN #{idx}] Title: {row['title']}")
    print(f"Company Profile (len={len(str(row['company_profile']))}): {str(row['company_profile'])[:120]}...")
    print(f"Description snippet: {str(row['description'])[:120]}...")
    print(f"Score: {row['decision_score']:.4f} (Threshold: {threshold})")
    print(f"Has logo: {row['has_company_logo']}, Telecommuting: {row['telecommuting']}")
