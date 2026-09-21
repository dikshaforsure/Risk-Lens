# 🛡️ Risk-Lens – Fraud Risk Scoring System

**Risk-Lens** is a machine learning-powered fraud detection system designed to identify suspicious credit card transactions through risk-aware scoring. Rather than providing a simple binary "yes/no," the system assigns a probability-based score to categorize transactions into **Low**, **Medium**, or **High** risk levels.

This project addresses core real-world ML challenges: **extreme class imbalance**, **cost-sensitive decision-making**, and **model explainability**.

---

## 🎯 Problem Statement
Credit card fraud is a rare but highly costly event. Traditional rule-based systems often struggle with evolving fraud patterns and high false-positive rates, leading to customer dissatisfaction.

* **Early Detection:** Identifying fraud before the transaction is finalized.
* **User Experience:** Minimizing friction for genuine users.
* **Explainability:** Providing "Why" a transaction was flagged for manual review.

---

## 🧠 Solution Approach
Risk-Lens treats fraud detection as a **Risk Assessment** problem rather than a standard classification task.

1.  **Pattern Learning:** Trains on historical PCA-transformed transaction data.
2.  **Probability Mapping:** Outputs a raw fraud score between 0.0 and 1.0.
3.  **Risk Tiering:** Maps scores to actionable risk categories (Low, Medium, High).
4.  **Decision Logic:** Triggers specific system actions (Allow, Verify, or Flag) based on the tier.



---

## 📊 Dataset Overview
The system utilizes the **Credit Card Fraud Detection Dataset** (European cardholders).

* **Total Transactions:** 284,807
* **Fraud Cases:** 492 (≈ 0.17%) — *Extreme Class Imbalance*
* **Features:** `Time`, `Amount`, and `V1–V28` (Anonymized PCA components).
* **Target:** `Class` (1 = Fraud, 0 = Genuine).

---

## 🏗️ System Architecture
The pipeline ensures a smooth flow from raw data to a final business decision:

`Transaction Data` ➔ `Preprocessing & Scaling` ➔ `Machine Learning Model` ➔ `Risk Scoring` ➔ `Decision Logic`

---

## 🧪 Machine Learning Pipeline

### 1. Data Preprocessing
* **Stratified Splitting:** Ensures the 0.17% fraud distribution is maintained in both training and test sets.
* **Standardization:** Features are scaled using `StandardScaler`.
* **Imbalance Handling:** Utilizes **Class-Weighted Learning** (Balanced Logistic Regression) to ensure the model pays more attention to the minority fraud class.

### 2. Model Selection
**Balanced Logistic Regression** was selected as the final model due to:
* High **ROC-AUC** performance.
* Exceptional **interpretability** (crucial for financial auditing).
* Stability on PCA-transformed data.

---

## ⚖️ Risk Scoring Logic

| Risk Score | Risk Level | System Decision |
| :--- | :--- | :--- |
| `< 0.20` | 🟢 **LOW** | **Allow** transaction automatically |
| `0.20 – 0.70` | 🟡 **MEDIUM** | **Require Verification** (MFA/OTP) |
| `> 0.70` | 🔴 **HIGH** | **Flag for Review** by Analyst |

---

## 🔍 Explainability
Risk-Lens prioritizes "White-box" AI:
* **Global Interpretability:** Logistic Regression coefficients reveal which internal signals most influence fraud.
* **Local Interpretability:** Integrated **SHAP values** explain why a specific transaction received a high score.

---

## 🖥️ User Interface
A lightweight **Streamlit** dashboard allows users to interact with the engine.
* **Demo Mode:** Test with pre-filled transaction signals.
* **Manual Input:** Input specific feature values to observe score changes.
* **Visual Feedback:** Real-time display of risk level and recommended action.

---

## 📁 Project Structure
```text
Risk-Lens/
├── app.py                # Streamlit UI
├── src/
│   ├── preprocess.py     # Data loading & scaling
│   ├── train.py          # Model training & saving
│   ├── evaluate.py       # Evaluation utilities
│   └── risk_logic.py     # Risk & decision logic
├── data/
│   └── raw/              # creditcard.csv
├── results/
│   ├── final_model.pkl   # Saved weights
│   └── scaler.pkl        # Saved scaler
├── notebooks/            # EDA & experimentation
├── requirements.txt      
└── README.md
```

---
## 🚀 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python src/train.py
```

### Launch the UI

```bash
streamlit run app.py
```

---
## 🏁 Conclusion
Risk-Lens demonstrates a production-ready approach to fraud detection by combining statistical learning with risk-aware logic. It moves beyond "Accuracy" to focus on Recall and Precision, ensuring that high-value fraud is captured while legitimate customers remain unaffected.
