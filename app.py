import os
import re
import joblib
from bs4 import BeautifulSoup
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 1. TEXT CLEANING FUNCTION (EXACT SAME AS TRAINING)
# ============================================================

def clean_text(text):
    """
    Applies the exact same text preprocessing pipeline used during training:
    1. Removes HTML markup using BeautifulSoup
    2. Converts text to lowercase
    3. Strips web URLs
    4. Removes non-alphabetic characters
    5. Collapses multiple whitespace into a single space
    """
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 2. CACHED MODEL LOADING
# ============================================================

@st.cache_resource
def load_artifacts():
    models_dir = "models"
    model_path = os.path.join(models_dir, "adaboost_model.pkl")
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    threshold_path = os.path.join(models_dir, "threshold.pkl")

    if not (os.path.exists(model_path) and os.path.exists(tfidf_path) and os.path.exists(threshold_path)):
        return None, None, None

    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    threshold = joblib.load(threshold_path)
    return model, tfidf, threshold


model, tfidf, threshold = load_artifacts()


# ============================================================
# 3. SIDEBAR: MODEL INFO & SAMPLE DATA
# ============================================================

with st.sidebar:
    st.title("🛡️ Model Overview")
    st.markdown(
        """
        This application detects fraudulent job advertisements using a machine-learning 
        pipeline trained on the **Employment Scam Aegean Dataset (EMSCAD)**.
        """
    )

    st.markdown("---")
    st.subheader("⚙️ Technical Specifications")
    st.markdown(
        f"""
        - **Algorithm**: AdaBoost Classifier
        - **Base Estimator**: Decision Tree (max_depth=4)
        - **Number of Estimators**: 100
        - **Learning Rate**: 0.5
        - **Feature Extraction**: TF-IDF (1-2 ngrams, 5000 features)
        - **Class Balancing**: Balanced sample weights
        - **Decision Threshold**: `{threshold if threshold is not None else 0.2:.2f}`
        - **Test F1 (Fraudulent)**: `78.11%`
        - **Test Precision (Fraudulent)**: `93.55%`
        """
    )

    st.markdown("---")
    st.subheader("💡 Load Example Inputs")

    if "job_title" not in st.session_state:
        st.session_state.job_title = ""
    if "company_profile" not in st.session_state:
        st.session_state.company_profile = ""
    if "job_desc" not in st.session_state:
        st.session_state.job_desc = ""
    if "requirements" not in st.session_state:
        st.session_state.requirements = ""
    if "benefits" not in st.session_state:
        st.session_state.benefits = ""

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Load Legitimate", use_container_width=True):
            st.session_state.job_title = "Senior Python Backend Engineer"
            st.session_state.company_profile = (
                "Acme Tech is a leading cloud software provider building scalable "
                "distributed systems since 2012 with 500+ employees globally."
            )
            st.session_state.job_desc = (
                "We are seeking an experienced Backend Engineer to design and build "
                "resilient microservices using Python, FastAPI, and PostgreSQL. You will "
                "collaborate with product managers and cross-functional engineering teams."
            )
            st.session_state.requirements = (
                "5+ years software development experience. Strong knowledge of Python, "
                "RESTful APIs, Docker, and CI/CD pipelines. BS in Computer Science or equivalent."
            )
            st.session_state.benefits = (
                "Competitive salary, comprehensive health insurance, 401(k) matching, "
                "and 20 days paid vacation per calendar year."
            )
            st.rerun()

    with col_btn2:
        if st.button("Load Suspicious", use_container_width=True):
            st.session_state.job_title = "Data Entry Clerk - Immediate Start - Earn $4000/week"
            st.session_state.company_profile = (
                "Global Financial Partners is rapidly expanding worldwide with immediate positions open."
            )
            st.session_state.job_desc = (
                "No experience needed! Work from home 2 hours per day. High pay guaranteed. "
                "Payment via wire transfer or cashier check. Must have bank account. Urgent hiring!"
            )
            st.session_state.requirements = (
                "Basic typing skills. Must be 18+ and ready to start immediately. Send resume "
                "and personal contact details to urgent-hiring-now@gmail.com."
            )
            st.session_state.benefits = (
                "Flexible hours, immediate cash bonuses, no background check required."
            )
            st.rerun()

    if st.button("Clear Form", use_container_width=True):
        st.session_state.job_title = ""
        st.session_state.company_profile = ""
        st.session_state.job_desc = ""
        st.session_state.requirements = ""
        st.session_state.benefits = ""
        st.rerun()


# ============================================================
# 4. MAIN USER INTERFACE
# ============================================================

st.title("Fake Job Posting Detector")
st.markdown(
    "Analyze job postings in real time to assess whether a posting matches linguistic "
    "patterns of legitimate or fraudulent recruitments."
)

if model is None or tfidf is None or threshold is None:
    st.error(
        "⚠️ Model artifacts not found in the `models/` directory. "
        "Please execute `python train_model.py` to train and save the model before using the application."
    )
    st.stop()

st.subheader("📋 Enter Job Details")

with st.container():
    title_input = st.text_input(
        "Job Title",
        value=st.session_state.job_title,
        placeholder="e.g., Software Engineer, Customer Service Representative"
    )

    company_profile_input = st.text_area(
        "Company Profile",
        value=st.session_state.company_profile,
        height=90,
        placeholder="Brief overview of the hiring company..."
    )

    description_input = st.text_area(
        "Job Description",
        value=st.session_state.job_desc,
        height=130,
        placeholder="Responsibilities, daily tasks, role overview..."
    )

    col1, col2 = st.columns(2)
    with col1:
        requirements_input = st.text_area(
            "Requirements",
            value=st.session_state.requirements,
            height=110,
            placeholder="Required skills, education, experience, qualifications..."
        )
    with col2:
        benefits_input = st.text_area(
            "Benefits",
            value=st.session_state.benefits,
            height=110,
            placeholder="Compensation, healthcare, insurance, leave policy..."
        )

# Analyze button
analyze_clicked = st.button("🔍 Analyze Job Posting", type="primary", use_container_width=True)


# ============================================================
# 5. PREDICTION & RESULTS DISPLAY
# ============================================================

if analyze_clicked:
    # 1. Combine fields in exact same order as training
    raw_combined = " ".join([
        title_input.strip(),
        company_profile_input.strip(),
        description_input.strip(),
        requirements_input.strip(),
        benefits_input.strip()
    ]).strip()

    # Handle empty input case gracefully
    if not raw_combined:
        st.warning(
            "⚠️ All fields are currently empty. Please provide at least a Job Title or "
            "Job Description to run the analysis."
        )
    else:
        # 2. Apply preprocessing
        cleaned_text = clean_text(raw_combined)

        if not cleaned_text:
            st.warning("⚠️ The input text contains no valid alphabetic content after cleaning.")
        else:
            with st.spinner("Analyzing text patterns with AdaBoost ensemble..."):
                # 3. Transform using fitted TF-IDF
                features = tfidf.transform([cleaned_text])

                # 4. AdaBoost raw decision function score (NOT a probability)
                decision_score = float(model.decision_function(features)[0])

                # 5. Compare with tuned validation threshold
                is_fraudulent = decision_score >= threshold

            st.markdown("---")
            st.subheader("📊 Analysis Results")

            # Result announcement
            if is_fraudulent:
                st.error("### Prediction: Potentially Fraudulent Job")
                st.markdown(
                    "The textual features of this posting match patterns frequently observed "
                    "in fraudulent listings within the training data."
                )
            else:
                st.success("### Prediction: Genuine Job")
                st.markdown(
                    "The textual features of this posting align with standard patterns observed "
                    "in legitimate job listings within the training data."
                )

            # Metric cards
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(
                    label="Model Decision Score",
                    value=f"{decision_score:+.4f}",
                    help="Signed distance from the ensemble decision boundary. Positive values indicate higher scam association."
                )
            with col_m2:
                st.metric(
                    label="Classification Threshold",
                    value=f"{threshold:.4f}",
                    help="Calibrated threshold selected on the validation set to balance precision (93.55%) and recall (67.05%)."
                )
            with col_m3:
                margin = decision_score - threshold
                st.metric(
                    label="Margin (Score - Threshold)",
                    value=f"{margin:+.4f}",
                    delta=f"{margin:+.4f}" if is_fraudulent else f"{margin:+.4f}",
                    delta_color="inverse"
                )

            # Important technical explanation & disclaimer
            st.info(
                f"""
                **Decision Rule Explanation**:
                - The AdaBoost model computes a continuous **decision score** by summing weighted predictions from 100 decision tree stumps/subtrees.
                - Unlike Logistic Regression, this decision score is **not a probability**; it represents the ensemble's signed margin.
                - A calibrated decision threshold of **`{threshold:.2f}`** is applied:
                  - If `Decision Score >= {threshold:.2f}` → Classified as **Potentially Fraudulent Job**
                  - If `Decision Score < {threshold:.2f}` → Classified as **Genuine Job**
                """
            )

            st.caption(
                "📌 **Disclaimer**: This prediction is generated by a statistical machine-learning model "
                "trained on historical postings. It is designed to assist review and does not constitute absolute legal or investigative proof. "
                "Always verify recruiter credentials and corporate domains independently before taking action."
            )
