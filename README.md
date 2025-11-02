# 🚨 Fake Job Detection & Analysis

## 🔍 Overview
This project detects and analyzes **fake job postings** using **Machine Learning**.  
It combines **data scraping**, **feature engineering**, **model training**, and **interactive visualization** using **Streamlit**.

Users can input a job description to predict whether it is **genuine or fake**, with explainable insights powered by **SHAP visualizations**.

---

## ✨ Features
✅ Fake Job Detection using supervised ML algorithms  
✅ Interactive Streamlit web app  
✅ SHAP-based model interpretability  
✅ Comprehensive performance metrics:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Confusion Matrix  
✅ Web scraping with **BeautifulSoup**  
✅ Automated **data preprocessing** and **feature engineering**

---

## 🛠 Installation

To run this project locally:

```bash
git clone https://github.com/Kriti4242/fake-job-posting-detector.git
cd fake-job-posting-detector
pip install -r requirements.txt
streamlit run main.py
 ```                                                                                                              
 ## 🔄 Project Workflow                
 Data Collection → Data Cleaning → Feature Engineering → Model Training   
        ↓  
   Model Evaluation → SHAP Explainability → Streamlit Deployment                                                      📊 Evaluation Metrics

Accuracy – Measures overall correctness

Precision – Measures true positive rate

Recall – Measures model sensitivity

F1 Score – Harmonic mean of precision and recall

ROC-AUC – Measures discrimination ability

Confusion Matrix – Visualizes true vs predicted classes

📚 Libraries Used

Streamlit – Interactive dashboard

Pandas, NumPy – Data handling and computation

Scikit-learn – Machine learning models and evaluation

SHAP, Matplotlib, Seaborn – Model explainability and visualization

BeautifulSoup4, Requests – Web scraping and data extraction

Joblib – Model serialization and persistence

⚖ License

This project is licensed under the MIT License.
You are free to use and modify it with proper attribution.


---             
 
