import re
import os
import joblib
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/DataSet.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. SELECT TEXT COLUMNS
# ============================================================

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

df["combined_text"] = (
    df[text_columns]
    .fillna("")
    .agg(" ".join, axis=1)
)


# ============================================================
# 3. CLEAN TEXT
# ============================================================

def clean_text(text):

    text = BeautifulSoup(
        text,
        "html.parser"
    ).get_text(" ")

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["clean_text"] = df["combined_text"].apply(
    clean_text
)


# ============================================================
# 4. CONVERT TARGET
# ============================================================

df["fraudulent"] = df["fraudulent"].map({
    "f": 0,
    "t": 1
})


print("\nTarget distribution:")
print(df["fraudulent"].value_counts())

print("\nTarget percentage:")
print(
    df["fraudulent"].value_counts(
        normalize=True
    ) * 100
)


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

X = df["clean_text"]
y = df["fraudulent"]


# ============================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)


print("\nData split:")
print("Training:", len(X_train_new))
print("Validation:", len(X_val))
print("Final Test:", len(X_test))


print("\nTraining fraud distribution:")
print(y_train_new.value_counts())

print("\nValidation fraud distribution:")
print(y_val.value_counts())

print("\nTest fraud distribution:")
print(y_test.value_counts())


# ============================================================
# 7. TF-IDF VECTORIZATION
# ============================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)


# Fit ONLY on training data
X_train_tfidf = tfidf.fit_transform(
    X_train_new
)

# Transform validation and test data
X_val_tfidf = tfidf.transform(
    X_val
)

X_test_tfidf = tfidf.transform(
    X_test
)


print("\nTF-IDF:")
print(
    "Training shape:",
    X_train_tfidf.shape
)

print(
    "Validation shape:",
    X_val_tfidf.shape
)

print(
    "Test shape:",
    X_test_tfidf.shape
)


# ============================================================
# 8. HANDLE CLASS IMBALANCE
# ============================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_new
)


print("\nSample weights:")
print(
    "Minimum:",
    sample_weights.min()
)

print(
    "Maximum:",
    sample_weights.max()
)


# ============================================================
# 9. BASE DECISION TREE
# ============================================================

base_estimator = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)


# ============================================================
# 10. ADABOOST MODEL
# ============================================================

model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)


# ============================================================
# 11. TRAIN ADABOOST
# ============================================================

model.fit(
    X_train_tfidf,
    y_train_new,
    sample_weight=sample_weights
)


print("\n")
print("=" * 60)
print("ADABOOST MODEL TRAINED")
print("=" * 60)

print("Base estimator : Decision Tree")
print("Tree max depth : 4")
print("Estimators     : 100")
print("Learning rate  : 0.5")
print("Class balancing: Enabled")


# ============================================================
# 12. VALIDATION DECISION SCORES
# ============================================================

validation_scores = model.decision_function(
    X_val_tfidf
)


# ============================================================
# 13. THRESHOLD SEARCH
# ============================================================

thresholds = [
    -0.4,
    -0.3,
    -0.2,
    -0.1,
     0.0,
     0.1,
     0.2,
     0.3,
     0.4
]


best_threshold = 0
best_f1 = 0


print("\n")
print("=" * 60)
print("VALIDATION THRESHOLD SEARCH")
print("=" * 60)

for threshold in thresholds:

    y_val_pred = (
        validation_scores >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:>4} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


print("-" * 60)

print(
    f"Best threshold: {best_threshold}"
)

print(
    f"Best validation F1: {best_f1:.4f}"
)


# ============================================================
# 14. FINAL TEST PREDICTION
# ============================================================

test_scores = model.decision_function(
    X_test_tfidf
)


y_test_pred = (
    test_scores >= best_threshold
).astype(int)


# ============================================================
# 15. FINAL TEST METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)


print("\n")
print("=" * 60)
print("FINAL TEST RESULTS - ADABOOST")
print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "Genuine",
            "Fraudulent"
        ],
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)


print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 18. CONFUSION MATRIX BREAKDOWN
# ============================================================

tn, fp, fn, tp = cm.ravel()


print("\nConfusion Matrix Breakdown:")

print(
    f"True Negatives  : {tn}"
)

print(
    f"False Positives : {fp}"
)

print(
    f"False Negatives : {fn}"
)

print(
    f"True Positives   : {tp}"
)


# ============================================================
# 19. FRAUD DETECTION SUMMARY
# ============================================================

print("\nFraud Detection Summary:")

print(
    f"Actual fraudulent jobs : {(y_test == 1).sum()}"
)

print(
    f"Detected fraudulent jobs: {tp}"
)

print(
    f"Missed fraudulent jobs  : {fn}"
)

print(
    f"Incorrectly flagged genuine jobs: {fp}"
)


# ============================================================
# 20. CREATE MODELS FOLDER
# ============================================================

models_folder = "models"

os.makedirs(
    models_folder,
    exist_ok=True
)


# ============================================================
# 21. SAVE ADABOOST MODEL
# ============================================================

model_path = os.path.join(
    models_folder,
    "adaboost_model.pkl"
)

joblib.dump(
    model,
    model_path
)


# ============================================================
# 22. SAVE TF-IDF VECTORIZER
# ============================================================

tfidf_path = os.path.join(
    models_folder,
    "tfidf_vectorizer.pkl"
)

joblib.dump(
    tfidf,
    tfidf_path
)


# ============================================================
# 23. SAVE THRESHOLD
# ============================================================

threshold_path = os.path.join(
    models_folder,
    "threshold.pkl"
)

joblib.dump(
    best_threshold,
    threshold_path
)


# ============================================================
# 24. VERIFY SAVED FILES
# ============================================================

print("\n")
print("=" * 60)
print("MODEL FILES SAVED")
print("=" * 60)

print(
    f"Model     : {model_path}"
)

print(
    f"TF-IDF    : {tfidf_path}"
)

print(
    f"Threshold : {threshold_path}"
)


print("\n")
print("=" * 60)
print("PROJECT TRAINING COMPLETE")
print("=" * 60)

print(
    f"Final Fraudulent F1: {f1:.4f}"
)

print(
    f"Final Threshold: {best_threshold}"
)