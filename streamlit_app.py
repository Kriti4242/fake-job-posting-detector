# -------------------- Imports --------------------
import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.exceptions import NotFittedError
from wordcloud import WordCloud

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="🚨 Fake Job Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom css
st.markdown("""
    <style>
    /* Vibrant Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #E0E0E0;
    }

    /* Headings with Neon Glow */
    h1, h2, h3, h4 {
        color: #FF6F00 !important;
        text-shadow: 0px 0px 8px rgba(255, 111, 0, 0.7);
        font-weight: bold;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1c1c1c, #2e2e2e);
        color: #FFFFFF;
    }

    /* Metric Cards */
    .stMetric {
        background: #1e293b;
        padding: 12px;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0,255,255,0.3);
        color: #00E5FF;
    }

    /* Buttons - Vibrant Orange */
    div.stButton > button {
        background: linear-gradient(90deg, #FF6F00, #FF4081);
        color: white;
        border-radius: 12px;
        font-weight: bold;
        padding: 10px 20px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #FF4081, #00E5FF);
        transform: scale(1.07);
    }

    /* Download Button */
    .stDownloadButton>button {
        background: linear-gradient(90deg, #00E5FF, #FF4081);
        color: white;
        border-radius: 12px;
        font-weight: bold;
        padding: 10px 20px;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #FF6F00, #00E5FF);
        transform: scale(1.07);
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
    }
    .real {
        background: linear-gradient(90deg, #28a745, #00c851);
        text-shadow: 0px 0px 6px rgba(0,0,0,0.6);
    }
    .fake {
        background: linear-gradient(90deg, #dc3545, #ff4444);
        text-shadow: 0px 0px 6px rgba(0,0,0,0.6);
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- Sidebar --------------------
st.sidebar.title("🚀 Fake Job Detector")
st.sidebar.markdown("**Go to Section:**")
st.sidebar.markdown("[🔎 Single Job Prediction](#single-job-prediction)")
st.sidebar.markdown("[📂 Batch Prediction](#batch-prediction)")
st.sidebar.markdown("[📖 Instructions](#instructions)")
st.sidebar.markdown("[📊 Model Info](#model-information--performance)")

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return joblib.load("fake_job_detector.joblib")

model_bundle = load_model()
model = model_bundle["model"]
vectorizer = model_bundle["vectorizer"]

# -------------------- Suspicious Keywords & Skills --------------------
SUSPICIOUS_KEYWORDS = ["urgent", "free laptop", "no experience needed", "immediate join", "work from home", "earn money"]
SKILL_LIST = [
    "python", "java", "c++", "sql", "excel", "power bi", "tableau",
    "machine learning", "deep learning", "nlp", "r", "html", "css",
    "javascript", "react", "aws", "docker", "kubernetes", "linux"
]

# -------------------- Helper Functions --------------------
def fetch_job_details(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200: return None
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.find("h1") or soup.find("h2") or soup.title)
        company = soup.select_one("div.company, span.companyName, span.topcard__org-name")
        location = soup.select_one("div.location, span.job-location, span.topcard__flavor.topcard__flavor--bullet")
        description = soup.select_one("div.jobsearch-jobDescriptionText, div.jd-desc, div#JobDescription, section.description")
        return {
            "job_title": title.get_text(strip=True) if title else "Not found",
            "company": company.get_text(strip=True) if company else "Not found",
            "location": location.get_text(strip=True) if location else "Not found",
            "description": description.get_text(separator=" ", strip=True) if description else None
        }
    except: 
        return None

def job_metrics(text: str):
    word_count = len(text.split())
    suspicious_count = sum(text.lower().count(k) for k in SUSPICIOUS_KEYWORDS)
    return word_count, suspicious_count

def extract_skills(text):
    text_lower = text.lower()
    return [skill for skill in SKILL_LIST if skill in text_lower]

def check_email_phone(text):
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", text)
    phones = re.findall(r"\+?\d[\d -]{8,}\d", text)
    email_flag = any("gmail.com" in e or "yahoo.com" in e or "hotmail.com" in e for e in emails)
    phone_flag = len(phones)==0
    return emails, phones, email_flag, phone_flag

# -------------------- Safe Predict Function --------------------
def predict(text):
    try:
        if not hasattr(vectorizer, "idf_"):
            st.error("⚠️ The TF-IDF vectorizer is not fitted. Please retrain the model using main.py and save the bundle again.")
            return "Error", 0.0

        features = vectorizer.transform([text])
        proba = model.predict_proba(features)[0,1]
        pred = "Fake" if proba>0.5 else "Real"
        return pred, proba
    except NotFittedError:
        st.error("⚠️ The TF-IDF vectorizer is not fitted. Please retrain the model using main.py and save the bundle again.")
        return "Error", 0.0
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        return "Error", 0.0

@st.cache_resource
def get_explainer(_model):
    return shap.Explainer(_model)
explainer = get_explainer(model)

# -------------------- Model Performance --------------------
X_test = model_bundle.get("X_test")
y_test = model_bundle.get("y_test")
preds_test = model.predict(X_test)
probs_test = model.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, preds_test)
pr = precision_score(y_test, preds_test, zero_division=0)
rc = recall_score(y_test, preds_test, zero_division=0)
f1 = f1_score(y_test, preds_test, zero_division=0)
roc = roc_auc_score(y_test, probs_test)
cm = confusion_matrix(y_test, preds_test)

# -------------------- Single Job Prediction --------------------
st.markdown("<a id='single-job-prediction'></a>", unsafe_allow_html=True)
st.header("🔎 Single Job Prediction")
job_input = st.text_area("Paste job description OR job URL:")

if st.button("Predict Single Job"):
    text, details = None, {}
    if job_input.startswith("http"):
        st.info("Fetching job details from URL...")
        details = fetch_job_details(job_input)
        if not details or not details.get("description"):
            st.warning("Could not fetch description. Please paste manually.")
            text = None
        else:
            text = details.get("description")
    else:
        text = job_input.strip()

    if text:
        pred, proba = predict(text)
        if pred == "Error":
            st.stop()

        real_pct = (1-proba)*100
        fake_pct = 100-real_pct

        # ---------- Prediction Metrics Cards ----------
        st.subheader("📌 Prediction Result")
        cols = st.columns(6)

        # Custom Prediction Card
        if pred == "Real":
            cols[0].markdown(f"<div class='prediction-card real'>🟢 REAL JOB</div>", unsafe_allow_html=True)
        else:
            cols[0].markdown(f"<div class='prediction-card fake'>🔴 FAKE JOB</div>", unsafe_allow_html=True)

        cols[1].metric("Real %", f"{real_pct:.2f}%")
        cols[2].metric("Fake %", f"{fake_pct:.2f}%")
        word_count, suspicious_count = job_metrics(text)
        skills = extract_skills(text)
        emails, phones, email_flag, phone_flag = check_email_phone(text)
        cols[3].metric("Word Count", word_count)
        cols[4].metric("Skills Detected", ', '.join(skills) if skills else "None")
        cols[5].metric("Emails Found", len(emails))
        st.markdown(f"**Phone Numbers Found:** {', '.join(phones) if phones else 'None'}")

        # ---------- Confidence Bar ----------
        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(["Real", "Fake"], [real_pct, fake_pct], color=["#0077b6","#ff6f00"])
        ax.set_ylim(0,100)
        ax.set_ylabel("Confidence (%)")
        for i, v in enumerate([real_pct, fake_pct]):
            ax.text(i, v+1, f"{v:.1f}%", ha="center", fontweight="bold")
        st.pyplot(fig)

        # ---------- Suspicious Reasons ----------
        st.subheader("🚨 Why this job may be suspicious?")
        reasons = []
        if suspicious_count>0: reasons.append(f"Suspicious keywords found ({suspicious_count})")
        if email_flag: reasons.append("Uses free email domain")
        if phone_flag: reasons.append("No phone number detected")
        if pred=="Fake": reasons.append("ML model predicts fake")
        if reasons:
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.markdown("No major suspicious indicators detected.")

        # ---------- Job Details ----------
        st.subheader("📝 Job Details")
        if details:
            cols = st.columns(3)
            cols[0].markdown(f"**Job Title:** {details['job_title']}")
            cols[1].markdown(f"**Company:** {details['company']}")
            cols[2].markdown(f"**Location:** {details['location']}")

        # ---------- SHAP Top Features ----------
        st.subheader("🔎 Top Features Influencing Prediction")
        try:
            dense_sample = vectorizer.transform([text]).toarray()
            shap_values = explainer(dense_sample)
            feature_names = vectorizer.get_feature_names_out()
            feature_importance = pd.DataFrame({
                "feature": feature_names,
                "importance": shap_values.values[0]
            }).sort_values("importance", key=abs, ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x="importance", y="feature", data=feature_importance, palette="Blues")
            ax.set_title("Top Features Influencing Prediction")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")

# -------------------- Batch Prediction --------------------
st.markdown("<a id='batch-prediction'></a>", unsafe_allow_html=True)
st.header("📂 Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV with 'description' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "description" not in df.columns:
        st.error("CSV must have 'description' column")
    else:
        st.info("Running batch predictions...")

        predictions, confidences, reasons_list, skills_list = [], [], [], []

        for text in df["description"].astype(str):
            pred, proba = predict(text)
            if pred == "Error":
                st.stop()
            real_pct = (1-proba)*100

            word_count, suspicious_count = job_metrics(text)
            skills = extract_skills(text)
            emails, phones, email_flag, phone_flag = check_email_phone(text)

            skills_list.append(", ".join(skills) if skills else "None")

            reasons = []
            if suspicious_count>0: reasons.append(f"{suspicious_count} suspicious keywords")
            if email_flag: reasons.append("Uses free email domain")
            if phone_flag: reasons.append("No phone number detected")
            if pred=="Fake": reasons.append("ML model predicts fake")
            reasons_list.append(", ".join(reasons) if reasons else "None")

            predictions.append(pred)
            confidences.append(real_pct)

        df["Prediction"] = predictions
        df["Confidence_Real_%"] = confidences
        df["Suspicious_Reasons"] = reasons_list
        df["Skills_Detected"] = skills_list

        # ---------- Display Sample ----------
        st.subheader("📊 Batch Prediction Sample")
        st.dataframe(df.head(10))

        # ---------- Pie Chart ----------
        fig, ax = plt.subplots()
        df_counts = df["Prediction"].value_counts()
        ax.pie(df_counts, labels=df_counts.index, autopct='%1.1f%%', colors=["#0077b6","#ff6f00"])
        ax.set_title("Real vs Fake Jobs Distribution")
        st.pyplot(fig)

        # ---------- WordCloud ----------
        all_text = " ".join(df["description"].astype(str).tolist()).lower()
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(all_text)
        st.subheader("WordCloud of Job Descriptions")
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt.gcf())

        # ---------- Download CSV ----------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Full Batch Results", csv, "batch_predictions.csv", "text/csv")

# -------------------- Instructions --------------------
st.markdown("<a id='instructions'></a>", unsafe_allow_html=True)
st.header("📖 Instructions")
st.markdown("""
- Paste a job URL or description in **Single Job Prediction**.
- Upload CSV in **Batch Prediction** to run multiple jobs.
- Charts and wordcloud use Blue–Orange theme for better visualization.
- Results can be downloaded.
""")

# -------------------- Model Info --------------------
st.markdown("<a id='model-information--performance'></a>", unsafe_allow_html=True)
st.header("📊 Model Information & Performance")
st.markdown(f"- Model: TF-IDF + XGBoost classifier")
st.markdown(f"- Accuracy: {acc*100:.2f}%")
st.markdown(f"- F1-score: {f1*100:.2f}%")
st.markdown(f"- Precision: {pr*100:.2f}%")
st.markdown(f"- Recall: {rc*100:.2f}%")
st.markdown(f"- ROC-AUC: {roc:.2f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Real","Fake"], yticklabels=["Real","Fake"], ax=ax)
st.pyplot(fig)

st.markdown("---")
st.markdown("✅ **This model is trained with supervised ML and TF-IDF features. Use results for evaluation purposes.**")
