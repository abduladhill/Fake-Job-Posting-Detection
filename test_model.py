import os
import re
import joblib
from bs4 import BeautifulSoup


# ============================================================
# 1. TEXT CLEANING FUNCTION (EXACT SAME AS TRAINING)
# ============================================================

def clean_text(text):
    """
    Cleans raw text by:
    1. Removing HTML tags using BeautifulSoup
    2. Lowercasing
    3. Removing URLs
    4. Removing non-alphabetic characters
    5. Normalizing whitespace
    """
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 2. LOAD SAVED MODEL ARTIFACTS
# ============================================================

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "adaboost_model.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.pkl")

print("=" * 65)
print("LOADING TRAINED MODEL ARTIFACTS")
print("=" * 65)

if not (os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH) and os.path.exists(THRESHOLD_PATH)):
    raise FileNotFoundError("Model artifacts not found! Run train_model.py first.")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
threshold = joblib.load(THRESHOLD_PATH)

print(f"Loaded Model     : {type(model).__name__}")
print(f"Loaded Vectorizer: {type(tfidf).__name__} (Vocabulary size: {len(tfidf.vocabulary_)})")
print(f"Loaded Threshold : {threshold}")
print("=" * 65)


# ============================================================
# 3. SAMPLE TEST POSTINGS
# ============================================================

sample_jobs = [
    {
        "label": "Legitimate Job Example (Senior Python Engineer)",
        "fields": {
            "title": "Senior Python Backend Engineer",
            "company_profile": "Acme Tech is a leading cloud software provider building scalable distributed systems since 2012.",
            "description": "We are seeking an experienced Backend Engineer to build resilient microservices using Python, FastAPI, and PostgreSQL.",
            "requirements": "5+ years software development experience. Strong knowledge of Python, RESTful APIs, Docker, and CI/CD pipelines. BS in Computer Science or equivalent.",
            "benefits": "Competitive salary, comprehensive health insurance, 401(k) matching, and 20 days paid vacation."
        }
    },
    {
        "label": "Suspicious / Scam Job Example (Urgent Data Entry Work From Home)",
        "fields": {
            "title": "Data Entry Clerk - Immediate Start - Earn $4000/week",
            "company_profile": "Global Financial Partners is rapidly expanding worldwide with immediate positions open.",
            "description": "No experience needed! Work from home 2 hours per day. High pay guaranteed. Payment via wire transfer or cashier check. Must have bank account.",
            "requirements": "Basic typing skills. Must be 18+ and ready to start immediately. Send resume and personal contact details to urgent-hiring@gmail.com.",
            "benefits": "Flexible hours, immediate cash bonuses, no background check required."
        }
    },
    {
        "label": "Edge Case Example (Minimal / Sparse Input)",
        "fields": {
            "title": "Sales Representative",
            "company_profile": "",
            "description": "Commission based sales representative wanted.",
            "requirements": "",
            "benefits": ""
        }
    }
]


# ============================================================
# 4. PREDICTION FUNCTION
# ============================================================

def predict_job(title="", company_profile="", description="", requirements="", benefits=""):
    # Combine fields in exact same order as training
    combined_raw = " ".join([
        title or "",
        company_profile or "",
        description or "",
        requirements or "",
        benefits or ""
    ])

    # Clean text
    cleaned = clean_text(combined_raw)

    # Transform using fitted TF-IDF
    features = tfidf.transform([cleaned])

    # AdaBoost raw decision score (NOT a probability)
    decision_score = float(model.decision_function(features)[0])

    # Compare against calibrated threshold
    is_fraudulent = int(decision_score >= threshold)

    prediction_label = (
        "Potentially Fraudulent Job (1)" if is_fraudulent == 1
        else "Genuine Job (0)"
    )

    return {
        "raw_text_length": len(combined_raw),
        "clean_text": cleaned[:120] + ("..." if len(cleaned) > 120 else ""),
        "decision_score": decision_score,
        "threshold": threshold,
        "prediction_label": prediction_label,
        "is_fraudulent": is_fraudulent
    }


# ============================================================
# 5. RUN INFERENCE ON SAMPLE POSTINGS
# ============================================================

for idx, sample in enumerate(sample_jobs, 1):
    print(f"\n--- Test Sample {idx}: {sample['label']} ---")
    fields = sample["fields"]
    res = predict_job(**fields)
    
    print(f"Title           : {fields['title']}")
    print(f"Cleaned Snippet : {res['clean_text']}")
    print(f"Decision Score  : {res['decision_score']:.4f}")
    print(f"Threshold       : {res['threshold']:.4f}")
    print(f"Condition Met   : Decision Score >= Threshold -> {res['decision_score'] >= res['threshold']}")
    print(f"Prediction      : {res['prediction_label']}")
    print("-" * 65)

print("\nImportant Note: The AdaBoost decision function outputs a real-valued score representing")
print("the signed margin/confidence from the boosting ensemble. It is NOT a calibrated probability.")
