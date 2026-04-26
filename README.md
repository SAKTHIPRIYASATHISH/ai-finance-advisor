# AI Finance Advisor

An end-to-end ML-powered personal finance app built for Indian users. Tracks UPI/card expenses, auto-classifies transactions, predicts next month's spending, detects anomalies, and delivers AI-generated budget advice — all in one Streamlit dashboard.

## Live Demo

sakthipriya2026-ai-finance-advisor.hf.space

## Features

| Feature | Tech | Status |
|---|---|---|
| Expense tracker with Indian merchants | Streamlit + SQLite | Week 1 |
| Auto-category classifier (NLP) | TF-IDF + Naive Bayes | Week 2 |
| Monthly spend predictor | Linear Regression | Week 2 |
| SHAP explainability | SHAP library | Week 2 |
| Anomaly detection | Z-score + Isolation Forest | Week 3 |
| AI budget advisor | Gemini API | Week 3 |

---

## ML Pipeline

```
New transaction description
        ↓
[Classifier] TF-IDF + Naive Bayes → predicted category
        ↓
[Predictor] Linear Regression on monthly history → next month forecast
        ↓
[SHAP] Feature importance → "why this prediction"
        ↓
[Anomaly Detector] Z-score → flags unusual spends
        ↓
[AI Advisor] Groq + LLaMA 3.3 → 3 natural language budget tips
```

---

## Tech Stack

- **Frontend/UI:** Streamlit
- **ML:** scikit-learn (Naive Bayes, Linear Regression, Isolation Forest)
- **Explainability:** SHAP
- **Data:** pandas, numpy
- **Charts:** Plotly
- **Database:** SQLite
- **AI Layer:** Groq + LLaMA 3.3
- **Deployment:** Hugging Face Spaces

---

## Run Locally

```bash
# 1. Clone and setup
git clone https://github.com/SAKTHIPRIYASATHISH/ai-finance-advisor
cd ai-finance-advisor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate Indian synthetic dataset
python generate_data.py

# 3. Run the app
streamlit run app/main.py
```

---

## Dataset

Synthetic dataset of 2000 Indian transactions covering:
- UPI merchants: Swiggy, Zomato, Ola, Uber, IRCTC, BigBasket, Blinkit
- Payment modes: UPI (50%), Credit Card (20%), Debit Card (15%), Net Banking, Cash
- Categories: Food, Transport, Shopping, Groceries, Bills, EMI, Health, Entertainment, Education, Personal Care
- Intentional anomalies injected for training the anomaly detector

---

## Model Performance

| Model | Metric | Score |
|---|---|---|
| Category Classifier | Accuracy | ~94% |
| Expense Predictor | R² Score | ~0.89 |
| Anomaly Detector | Precision | ~91% |

*(Update after training)*

---

Built by Sakthi Priya S
