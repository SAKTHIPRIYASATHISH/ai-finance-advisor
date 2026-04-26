# ─────────────────────────────────────────────────────────────
# advisor.py
# Uses Groq API (free, works in India) instead of Gemini
# Model: llama3-8b-8192 — fast and free
# ─────────────────────────────────────────────────────────────

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def build_prompt(financial_summary: dict) -> str:
    s = financial_summary

    cat_lines = "\n".join([
        f"  - {cat}: Rs.{amt:,.0f}"
        for cat, amt in sorted(
            s.get('category_breakdown', {}).items(),
            key=lambda x: x[1], reverse=True
        )
    ])

    anomaly_lines = ""
    if s.get('anomaly_details'):
        anomaly_lines = "\n".join([f"  - {a}" for a in s['anomaly_details'][:3]])
    else:
        anomaly_lines = "  - None detected"

    prompt = f"""
You are a personal finance advisor for an Indian user.
Analyse the following financial data and give exactly 3 specific, actionable budget tips.

FINANCIAL DATA FOR {s.get('month_label', 'this month')}

Total spent this month: Rs.{s.get('total_spent', 0):,.0f}
Top spending category: {s.get('top_category', 'N/A')} (Rs.{s.get('top_category_amt', 0):,.0f})

Category breakdown:
{cat_lines}

ML PREDICTION
Predicted spend next month: Rs.{s.get('predicted_next', 0):,.0f}
Expected range: Rs.{s.get('prediction_low', 0):,.0f} to Rs.{s.get('prediction_high', 0):,.0f}
Key driver of prediction: {s.get('shap_top_factor', 'N/A')} (impact: Rs.{s.get('shap_top_value', 0):,})

ANOMALIES DETECTED
Number of unusual transactions: {s.get('anomaly_count', 0)}
Total anomalous amount: Rs.{s.get('anomaly_amount', 0):,.0f}
Details:
{anomaly_lines}

YOUR TASK
Give exactly 3 budget tips. Each tip must:
1. Be specific to this user's actual numbers (mention amounts in Rs.)
2. Be actionable (tell them exactly what to do)
3. Be realistic for an Indian user (mention UPI, apps, Indian context)
4. Be 2-3 sentences maximum

Format your response exactly like this:
TIP 1: [title]
[advice]

TIP 2: [title]
[advice]

TIP 3: [title]
[advice]
"""
    return prompt.strip()


def get_ai_advice(financial_summary: dict) -> list:
    """
    Main function called by the app.
    Returns a list of 3 tips, each as:
      {'title': '...', 'advice': '...'}
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Make sure your .env file has: GROQ_API_KEY=your_key_here"
            )

        client = Groq(api_key=api_key)
        prompt = build_prompt(financial_summary)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7
        )

        text = response.choices[0].message.content.strip()
        return parse_tips(text)

    except Exception as e:
        return [{
            'title': 'API Error',
            'advice': f"Could not connect to Groq: {str(e)}. "
                      f"Check your GROQ_API_KEY in the .env file."
        }]


def parse_tips(text: str) -> list:
    tips = []
    lines = text.strip().split('\n')
    current_title = None
    current_advice = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith('TIP') and ':' in line:
            if current_title and current_advice:
                tips.append({
                    'title':  current_title,
                    'advice': ' '.join(current_advice).strip()
                })
                current_advice = []
            parts = line.split(':', 1)
            current_title = parts[1].strip().strip('*').strip() if len(parts) > 1 else line
        elif current_title:
            current_advice.append(line.strip('*').strip())

    if current_title and current_advice:
        tips.append({
            'title':  current_title,
            'advice': ' '.join(current_advice).strip()
        })

    if not tips:
        tips = [{'title': 'AI Advice', 'advice': text}]

    return tips[:3]


def build_summary_from_app(df_month, pred, anomaly_summary, shap_explanation, month_label):
    cat_breakdown = {}
    if not df_month.empty:
        cat_breakdown = df_month.groupby('category')['amount'].sum().to_dict()

    anomaly_details = []
    if anomaly_summary['total_flagged'] > 0:
        flagged = anomaly_summary['flagged_df']
        for _, row in flagged.head(3).iterrows():
            anomaly_details.append(row.get('anomaly_reason', row['description']))

    shap_top = shap_explanation[0] if shap_explanation else {}

    total = df_month['amount'].sum() if not df_month.empty else 0
    top_cat, top_amt = ('N/A', 0)
    if not df_month.empty:
        grp     = df_month.groupby('category')['amount'].sum()
        top_cat = grp.idxmax()
        top_amt = grp.max()

    return {
        'total_spent':        round(total),
        'top_category':       top_cat,
        'top_category_amt':   round(top_amt),
        'predicted_next':     pred['prediction'],
        'prediction_low':     pred['low'],
        'prediction_high':    pred['high'],
        'anomaly_count':      anomaly_summary['total_flagged'],
        'anomaly_amount':     round(anomaly_summary['total_amount']),
        'anomaly_details':    anomaly_details,
        'shap_top_factor':    shap_top.get('feature', 'N/A'),
        'shap_top_value':     abs(shap_top.get('shap_value', 0)),
        'category_breakdown': cat_breakdown,
        'month_label':        month_label,
    }


if __name__ == '__main__':
    print("Testing Groq API connection...")
    dummy = {
        'total_spent':        48500,
        'top_category':       'EMI & Finance',
        'top_category_amt':   18200,
        'predicted_next':     52000,
        'prediction_low':     46800,
        'prediction_high':    57200,
        'anomaly_count':      2,
        'anomaly_amount':     9800,
        'anomaly_details':    [
            'Rs.6200 Zomato order - 540% above normal Food spend',
            'Rs.3600 Ola outstation - 280% above normal Transport spend'
        ],
        'shap_top_factor':    '3-month average',
        'shap_top_value':     12400,
        'category_breakdown': {
            'Food': 8200, 'Transport': 5400, 'Shopping': 7100,
            'EMI & Finance': 18200, 'Entertainment': 3100,
            'Groceries': 4200, 'Health': 2300
        },
        'month_label': 'April 2024'
    }
    tips = get_ai_advice(dummy)
    print(f"\nGot {len(tips)} tips:\n")
    for i, tip in enumerate(tips, 1):
        print(f"TIP {i}: {tip['title']}")
        print(f"  {tip['advice']}\n")