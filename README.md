# 📱 SMS Spam Detection with Explainable AI

## 🧠 Overview
This project presents a **real-time, explainable SMS spam detection system** that combines **machine learning** with **explainable AI (XAI)** techniques to achieve high accuracy while maintaining transparency and user trust.  
Built with **Python, Streamlit, and Docker**, the system detects spam messages and provides **LIME** and **SHAP** explanations to clarify why each message was classified as spam or ham.

---

## 🚨 Problem Statement
- Rising global spam and phishing threats in 2025  
  - 💰 **$16.2B** projected global mobile SMS scam losses ([Kaspersky, 2025])  
- **Rule-based filters** fail against adaptive spam patterns  
- **Black-box AI models**: high accuracy but no transparency  
- **Dual Failure Modes**:
  - ❌ *False Positives:* Legitimate messages (OTPs, alerts) blocked  
  - ⚠️ *False Negatives:* Spam with phishing/malware slips through  
- **Regulatory Compliance (GDPR Article 22):** Automated decisions must be explainable  

---

## 🎯 Objectives
1. Develop a **high-precision SMS spam detection model**
2. Ensure **interpretability and explainability**
3. Reduce **false positives** and **false negatives**
4. Enable **real-time spam prediction**
5. Support **scalable deployment** using Docker

---

## ⚙️ Methodology
### High-Level Architecture
1. **User Input:** SMS message  
2. **Preprocessing Pipeline:** Tokenization → Stopword Removal → Stemming → TF-IDF  
3. **Model:** Multinomial Naive Bayes  
4. **Explainability Layer:**  
   - LIME (Local Explanation)  
   - SHAP (Local + Global Explanation)  
   - Hybrid Local Explanation = LIME + SHAP  
5. **Output:** Prediction (Spam/Ham) + Explanations  
6. **Deployment:** Streamlit Web UI (Dockerized)

*(See presentation for architecture diagram.)*

---

## 📊 Dataset and Preprocessing
**Source:** [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
**Size:** 5,574 labeled SMS messages  
- Ham: 86.6%  
- Spam: 13.4%  

**Composition:**
- 425 spam messages – Grumbletext  
- 3,375 ham – NUS SMS Corpus  
- 450 ham – Caroline Tag’s PhD thesis  
- 1,324 messages (1,002 ham / 322 spam) – SMS Spam Corpus v0.1 Big  

**Preprocessing Steps:**
1. Convert to lowercase  
2. Tokenize (`\b\w+\b`)  
3. Remove stopwords and punctuation  
4. Apply Porter Stemming  
5. Vectorize using TF-IDF  

---

|

**Final Model:** Multinomial Naive Bayes  
- Accuracy: 96.3%  
- Precision: 98.1%  
- Recall: 73.9%  
- F1-Score: 84.3%  
- Confusion Matrix: TP=102, TN=894, FP=2, FN=36  

---

## 🔍 Explainability Techniques
### Why Explainability Matters
Black-box predictions reduce trust — users and regulators need transparency.

### Techniques Used
- **LIME (Local):** Highlights words influencing a single prediction (e.g., “free,” “claim”)  
- **SHAP (Local):** Quantifies each feature’s contribution for a specific SMS  
- **Hybrid Local Explanation:** Combines LIME + SHAP weights for more stable explanations  
- **SHAP (Global):** Shows overall feature importance across dataset  

**Benefits:**
- Improves user trust  
- Aids debugging  
- Ensures GDPR compliance  

---

## 🧰 Tools & Environment

| Tool | Purpose | Version |
|------|----------|---------|
| Python | Core language | 3.11.9 |
| scikit-learn | ML models | 1.5.0 |
| SHAP | Explainability | 0.44.0 |
| LIME | Local explanations | 0.2.0.1 |
| NLTK | Preprocessing | 3.8.1 |
| Streamlit | Web interface | 1.32.0 |
| Git | Version control | 2.43 |
| Docker | Deployment | 28.5.1 |

---

## 🧾 Key Findings
- Achieved **98% precision** using Multinomial Naive Bayes  
- Combined **LIME + SHAP** local explanations improved interpretability  
- Fully **reproducible**, **deployable**, and **web-based** system  

### Limitations
- English-only dataset  
- Dataset may not reflect modern (2025) spam tactics  
- SHAP computation latency in real-time applications  

### Future Work
- Adaptive learning for evolving spam tactics  
- Multilingual & multimodal detection (text, image, QR codes)

---


---

## 📚 References
1. Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. ACM DocEng '11.  
2. Kaspersky. (2025). *Global Mobile Scam Report 2025*.  
3. Lookout. (2025). *Consumer Mobile Security Report 2025*.  
4. Gartner. (2022). *Email and Messaging Security: Market Trends and User Pain Points*.  
5. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *“Why Should I Trust You?”: Explaining the Predictions of Any Classifier*. KDD '16.  
6. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NIPS 2017.  
7. Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.). Leanpub.  
8. European Union. (2016). *General Data Protection Regulation (GDPR), Article 22*.  

---