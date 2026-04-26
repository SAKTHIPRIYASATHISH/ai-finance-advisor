# ─────────────────────────────────────────────────────────────
# anomaly.py
# What this file does:
#   1. Learns the "normal" spending range per category (Z-score)
#   2. Flags transactions that are unusually high
#   3. Also trains an Isolation Forest (advanced ML method)
#   4. Returns flagged transactions with reason
#
# Z-score explained:
#   - Calculate mean and std of amounts per category
#   - Z-score = (amount - mean) / std
#   - Z-score > 2 means the amount is 2 standard deviations
#     above normal → flag it as anomaly
#   - Example: Food avg=₹350, std=₹200
#              Z-score of ₹4000 = (4000-350)/200 = 18.25 → ANOMALY
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'anomaly.pkl')
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.csv')

# Z-score threshold — transactions above this are flagged
Z_THRESHOLD = 2.0


def compute_category_stats(df):
    """
    For each category, compute mean and std of transaction amounts.
    This defines what 'normal' looks like per category.
    """
    stats = df.groupby('category')['amount'].agg(['mean', 'std']).reset_index()
    stats.columns = ['category', 'mean_amount', 'std_amount']
    # If std is 0 or NaN (only 1 transaction), use mean * 0.3 as fallback
    stats['std_amount'] = stats['std_amount'].fillna(stats['mean_amount'] * 0.3)
    stats.loc[stats['std_amount'] == 0, 'std_amount'] = stats.loc[stats['std_amount'] == 0, 'mean_amount'] * 0.3
    return stats


def train():
    """
    Train Isolation Forest on transaction data.
    Isolation Forest works differently from Z-score:
    - It randomly splits data and checks how quickly a point gets isolated
    - Anomalies get isolated faster (they're far from the cluster)
    - More powerful than Z-score, catches multi-dimensional patterns
    """
    df = pd.read_csv(DATA_PATH)

    # Features for Isolation Forest
    # We use amount + category encoded as number
    df['category_code'] = pd.Categorical(df['category']).codes
    X = df[['amount', 'category_code']].values

    model = IsolationForest(
        n_estimators=100,   # 100 trees in the forest
        contamination=0.02, # expect ~2% anomalies
        random_state=42
    )
    model.fit(X)

    # Compute category stats for Z-score method
    stats = compute_category_stats(df)
    category_codes = dict(enumerate(pd.Categorical(df['category']).categories))

    joblib.dump({
        'isolation_forest': model,
        'category_stats':   stats,
        'category_codes':   category_codes
    }, MODEL_PATH)
    print(f"Anomaly detector trained → {MODEL_PATH}")

    # Quick test
    test_predictions = model.predict(X)
    n_anomalies = (test_predictions == -1).sum()
    print(f"Isolation Forest flagged {n_anomalies} anomalies in training data")
    print(f"Z-score method will flag amounts > {Z_THRESHOLD} std deviations from category mean")
    return model, stats


def load_detector():
    """Load saved detector. Train if not found."""
    if not os.path.exists(MODEL_PATH):
        print("No anomaly model found. Training now...")
        train()
    return joblib.load(MODEL_PATH)


def detect_anomalies_zscore(df):
    """
    Z-score based detection — simple, interpretable, fast.
    Returns a copy of df with extra columns:
      - z_score      : how many std devs above mean this transaction is
      - is_flagged   : True/False
      - anomaly_reason: human-readable explanation
    """
    saved = load_detector()
    stats = saved['category_stats']

    # Merge stats into transactions
    df = df.copy()
    df = df.merge(stats, on='category', how='left')

    # Calculate Z-score for each transaction
    df['z_score'] = (df['amount'] - df['mean_amount']) / df['std_amount']
    df['is_flagged'] = df['z_score'] > Z_THRESHOLD

    # Build human-readable reason
    def build_reason(row):
        if not row['is_flagged']:
            return ""
        pct_above = ((row['amount'] - row['mean_amount']) / row['mean_amount']) * 100
        return (f"₹{row['amount']:,.0f} is {pct_above:.0f}% above normal "
                f"{row['category']} spend (avg ₹{row['mean_amount']:,.0f})")

    df['anomaly_reason'] = df.apply(build_reason, axis=1)
    return df


def detect_anomalies_isolation_forest(df):
    """
    Isolation Forest based detection — more advanced ML method.
    Returns same format as Z-score method.
    """
    saved = load_detector()
    model = saved['isolation_forest']

    df = df.copy()
    df['category_code'] = pd.Categorical(df['category']).codes
    X = df[['amount', 'category_code']].values

    # -1 means anomaly, 1 means normal
    predictions = model.predict(X)
    df['is_flagged_if'] = predictions == -1
    return df


def get_anomaly_summary(df):
    """
    Main function called by the app.
    Returns:
      - flagged_df   : only the anomalous transactions
      - total_flagged: count
      - total_amount : sum of flagged transaction amounts
      - by_category  : how many anomalies per category
    """
    result = detect_anomalies_zscore(df)
    flagged = result[result['is_flagged']].copy()
    flagged = flagged.sort_values('z_score', ascending=False)

    by_category = (flagged.groupby('category')
                          .size()
                          .reset_index(name='count')
                          .sort_values('count', ascending=False))

    return {
        'flagged_df':    flagged,
        'total_flagged': len(flagged),
        'total_amount':  flagged['amount'].sum(),
        'by_category':   by_category
    }


if __name__ == '__main__':
    print("Training anomaly detector...")
    train()

    print("\n── Testing on sample data ─────────────────────────")
    df = pd.read_csv(DATA_PATH)
    summary = get_anomaly_summary(df)
    print(f"Total anomalies detected : {summary['total_flagged']}")
    print(f"Total anomalous amount   : ₹{summary['total_amount']:,.0f}")
    print("\nTop flagged transactions:")
    cols = ['description', 'amount', 'category', 'anomaly_reason']
    print(summary['flagged_df'][cols].head(6).to_string())