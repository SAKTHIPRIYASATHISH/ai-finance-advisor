# ─────────────────────────────────────────────────────────────
# predictor.py
# What this file does:
#   1. Aggregates transactions into monthly totals
#   2. Engineers features (month number, rolling average, etc.)
#   3. Trains a Linear Regression model to predict next month
#   4. Uses SHAP to explain WHY the prediction was made
#   5. Saves model to models/predictor.pkl
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import shap

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'predictor.pkl')
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.csv')


def build_features(df):
    """
    Convert raw transactions into monthly features for the model.

    Features we create:
    - month_num      : 1, 2, 3 ... 24 (overall time index)
    - month_of_year  : 1–12 (January=1, December=12, captures seasonal patterns)
    - rolling_avg_3  : average of last 3 months spending
    - rolling_avg_6  : average of last 6 months spending
    - prev_month     : last month's total (most recent signal)
    - food_ratio     : what % of spending was Food last month
    - emi_ratio      : what % was EMI (usually fixed, strong predictor)
    """
    # Aggregate to monthly totals
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly = df.groupby('month')['amount'].sum().reset_index()
    monthly = monthly.sort_values('month').reset_index(drop=True)
    monthly['month_num'] = range(1, len(monthly) + 1)
    monthly['month_of_year'] = monthly['month'].dt.month

    # Rolling averages — smooth out spikes
    monthly['rolling_avg_3'] = monthly['amount'].rolling(3, min_periods=1).mean()
    monthly['rolling_avg_6'] = monthly['amount'].rolling(6, min_periods=1).mean()

    # Previous month's total
    monthly['prev_month'] = monthly['amount'].shift(1).fillna(monthly['amount'].mean())

    # Category ratios per month
    cat_monthly = df.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)
    cat_monthly = cat_monthly.div(cat_monthly.sum(axis=1), axis=0)  # convert to ratios

    for cat, col in [
        ('Food',          'food_ratio'),
        ('EMI & Finance', 'emi_ratio'),
        ('Shopping',      'shopping_ratio'),
        ('Transport',     'transport_ratio'),
    ]:
        if cat in cat_monthly.columns:
            monthly[col] = monthly['month'].map(cat_monthly[cat]).fillna(0)
        else:
            monthly[col] = 0

    return monthly


def train():
    """Train the predictor and save to predictor.pkl"""
    df = pd.read_csv(DATA_PATH)
    monthly = build_features(df)

    FEATURE_COLS = [
        'month_num', 'month_of_year', 'rolling_avg_3',
        'rolling_avg_6', 'prev_month',
        'food_ratio', 'emi_ratio', 'shopping_ratio', 'transport_ratio'
    ]

    # Drop first row (no prev_month) and last row (that's what we want to predict)
    train_data = monthly.iloc[1:-1].copy()

    X = train_data[FEATURE_COLS]
    y = train_data['amount']

    # Build and train pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])
    model.fit(X, y)

    # Evaluate on training data
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    print(f"\nPredictor trained successfully")
    print(f"  R² score : {r2:.3f}  (1.0 = perfect, >0.8 = good)")
    print(f"  MAE      : ₹{mae:,.0f}  (average prediction error)")

    # Save model + feature columns + monthly data
    joblib.dump({
        'model': model,
        'feature_cols': FEATURE_COLS,
        'monthly': monthly
    }, MODEL_PATH)
    print(f"  Saved → {MODEL_PATH}")
    return model, monthly, FEATURE_COLS


def load_predictor():
    """Load saved predictor. Train if not found."""
    if not os.path.exists(MODEL_PATH):
        print("No predictor found. Training now...")
        train()
    return joblib.load(MODEL_PATH)


def predict_next_month():
    """
    Predict next month's total spending.
    Returns: dict with prediction, confidence range, and last 6 months history
    """
    saved   = load_predictor()
    model   = saved['model']
    fcols   = saved['feature_cols']
    monthly = saved['monthly']

    # Use the LAST row as input to predict the NEXT month
    last_row = monthly.iloc[-1][fcols].values.reshape(1, -1)
    prediction = model.predict(last_row)[0]

    # Confidence range: ±10% (simple heuristic)
    low  = prediction * 0.90
    high = prediction * 1.10

    # Last 6 months for chart
    history = monthly.tail(6)[['month', 'amount']].copy()
    history['month'] = history['month'].astype(str)

    return {
        'prediction': round(prediction),
        'low':        round(low),
        'high':       round(high),
        'history':    history
    }


def get_shap_explanation():
    """
    SHAP = SHapley Additive exPlanations
    Tells you WHY the model predicted that number.
    Example output:
      rolling_avg_3   +₹12,400  (3-month average is high → pushes prediction up)
      emi_ratio       +₹8,200   (EMI spend is consistent → adds to prediction)
      prev_month      -₹3,100   (last month was lower → pulls prediction down)
    """
    saved   = load_predictor()
    model   = saved['model']
    fcols   = saved['feature_cols']
    monthly = saved['monthly']

    # Background data for SHAP (use all training rows)
    X_bg   = monthly.iloc[1:-1][fcols].values
    X_last = monthly.iloc[-1][fcols].values.reshape(1, -1)

    # Scale the data the same way the pipeline does
    scaler    = model.named_steps['scaler']
    X_bg_sc   = scaler.transform(X_bg)
    X_last_sc = scaler.transform(X_last)

    # Create SHAP explainer on the linear regression step
    reg        = model.named_steps['reg']
    explainer  = shap.LinearExplainer(reg, X_bg_sc, feature_perturbation="interventional")
    shap_vals  = explainer.shap_values(X_last_sc)[0]

    # Build readable explanation
    explanation = []
    for feat, shap_val in zip(fcols, shap_vals):
        # Make feature names human-readable
        readable = (feat
            .replace('rolling_avg_3',       '3-month average')
            .replace('rolling_avg_6',       '6-month average')
            .replace('prev_month',          'last month spend')
            .replace('month_num',           'time trend')
            .replace('month_of_year',       'seasonal pattern')
            .replace('emi___finance_ratio', 'EMI proportion')
            .replace('food_ratio',          'Food proportion')
            .replace('shopping_ratio',      'Shopping proportion')
            .replace('transport_ratio',     'Transport proportion')
        )
        explanation.append({
            'feature':   readable,
            'shap_value': round(shap_val)
        })

    # Sort by absolute impact
    explanation.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    return explanation


if __name__ == '__main__':
    print("Training expense predictor...")
    train()

    print("\n── Prediction for next month ──────────────────────")
    result = predict_next_month()
    print(f"  Predicted spend : ₹{result['prediction']:,}")
    print(f"  Range           : ₹{result['low']:,} – ₹{result['high']:,}")

    print("\n── SHAP explanation (why this prediction?) ────────")
    for item in get_shap_explanation()[:5]:
        direction = "↑" if item['shap_value'] > 0 else "↓"
        print(f"  {direction} {item['feature']:<25} ₹{abs(item['shap_value']):,}")