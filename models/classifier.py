# ─────────────────────────────────────────────────────────────
# classifier.py
# What this file does:
#   1. Reads the transactions CSV
#   2. Converts description text → numbers (TF-IDF)
#   3. Trains a Naive Bayes model to predict category
#   4. Saves the trained model to models/classifier.pkl
#   5. Provides a predict_category() function for the app
# ─────────────────────────────────────────────────────────────

import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'classifier.pkl')
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.csv')


def train():
    """
    Train the classifier and save it to classifier.pkl
    Run this once: python -c "from models.classifier import train; train()"
    """
    # ── Step 1: Load data ──────────────────────────────────────
    df = pd.read_csv(DATA_PATH)

    # X = input  (the description text, e.g. "Swiggy order")
    # y = output (the category label, e.g. "Food")
    X = df['description']
    y = df['category']

    # ── Step 2: Split data into train and test sets ────────────
    # 80% of rows used for training, 20% held back for testing
    # random_state=42 means we get the same split every time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Step 3: Build the pipeline ────────────────────────────
    # A pipeline chains two steps:
    #   Step A — TfidfVectorizer: "Swiggy order" → [0.0, 0.82, 0.0, 0.45, ...]
    #   Step B — MultinomialNB:   numbers        → "Food"
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),   # consider single words AND two-word phrases
            lowercase=True,       # "Swiggy" and "swiggy" treated the same
            min_df=1              # include even rare words
        )),
        ('clf', MultinomialNB(alpha=0.1))
    ])

    # ── Step 4: Train ─────────────────────────────────────────
    # This is the actual "learning" step — takes < 1 second
    model.fit(X_train, y_train)

    # ── Step 5: Evaluate ──────────────────────────────────────
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {acc * 100:.1f}%")
    print("\nDetailed report:")
    print(classification_report(y_test, y_pred))

    # ── Step 6: Save the trained model ────────────────────────
    # joblib saves the entire trained model as a .pkl file
    # Later we load it without retraining every time
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    return model


def load_model():
    """Load the saved model. Train first if not found."""
    if not os.path.exists(MODEL_PATH):
        print("No saved model found. Training now...")
        return train()
    return joblib.load(MODEL_PATH)


def predict_category(description: str) -> str:
    """
    Given a transaction description, return the predicted category.

    Example:
        predict_category("Swiggy order")   → "Food"
        predict_category("Ola cab")        → "Transport"
        predict_category("Netflix")        → "Entertainment"
    """
    model = load_model()
    prediction = model.predict([description])
    return prediction[0]


def predict_with_confidence(description: str) -> dict:
    """
    Returns predicted category AND confidence percentage.
    Used by the app to show 'Food (94% confident)'
    """
    model = load_model()
    proba = model.predict_proba([description])[0]
    classes = model.classes_
    top_idx = proba.argmax()
    return {
        'category': classes[top_idx],
        'confidence': round(proba[top_idx] * 100, 1),
        'all_probs': dict(zip(classes, (proba * 100).round(1)))
    }


# ── Run this file directly to train and test ──────────────────
if __name__ == '__main__':
    print("Training classifier on Indian transaction data...")
    train()

    print("\n── Manual tests ──────────────────────────────────")
    test_cases = [
        "Swiggy order",
        "Ola cab",
        "Netflix subscription",
        "SBI home loan EMI",
        "Apollo pharmacy",
        "BigBasket order",
        "IRCTC train ticket",
        "Udemy course",
        "Salon haircut",
        "Flipkart order"
    ]
    for desc in test_cases:
        result = predict_with_confidence(desc)
        print(f"  '{desc}' → {result['category']} ({result['confidence']}%)")