🚨 Fake Job Detection & Analysis

🔍 Overview
This project detects and analyzes fake job postings using Machine Learning.
It combines data scraping, feature engineering, model training, and interactive visualization using Streamlit.

Users can input a job description to predict whether it is genuine or fake, with explainable insights via SHAP visualizations.

✨ Features

✅ Fake Job Detection using supervised ML algorithms

✅ Interactive Streamlit app

✅ SHAP visualizations for model interpretability

✅ Performance metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix

✅ Web scraping support with BeautifulSoup

✅ Data preprocessing and feature engineering


📂 Project Structure
project/
│
├── main.py                # Streamlit main app
├── model.pkl              # Pre-trained ML model
├── utils.py               # Helper functions for preprocessing and evaluation
├── data/
│   └── job_data.csv       # Dataset
└── README.md


🛠 Installation

Clone the repository:

git clone <repository-url>
cd project


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


🚀 Usage

Run the Streamlit application:

streamlit run main.py


Open the URL provided in your browser.

Input a job description to check if it is fake or genuine.

View SHAP visualizations for model explanation.


📊 Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

ROC-AUC Score

Confusion Matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


📚 Libraries Used

streamlit

pandas, numpy

scikit-learn

shap, matplotlib, seaborn

beautifulsoup4, requests

joblib


⚖ License


This project is licensed under the MIT License.
