import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import (init_db, add_transaction, get_all_transactions,
                             get_by_month, seed_from_csv, delete_transaction,
                             add_user, get_user, get_all_users)
from models.classifier import predict_with_confidence
from models.predictor  import predict_next_month, get_shap_explanation
from models.anomaly    import get_anomaly_summary
from utils.advisor     import get_ai_advice, build_summary_from_app

st.set_page_config(page_title="AI Finance Advisor", page_icon="₹", layout="wide")

init_db()
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.csv')

CATEGORIES = [
    "Food", "Transport", "Shopping", "Groceries",
    "Bills & Utilities", "EMI & Finance", "Health",
    "Entertainment", "Education", "Personal Care"
]
CATEGORY_COLORS = {
    "Food": "#1D9E75", "Transport": "#378ADD", "Shopping": "#D85A30",
    "Groceries": "#639922", "Bills & Utilities": "#534AB7",
    "EMI & Finance": "#D4537E", "Health": "#E24B4A",
    "Entertainment": "#BA7517", "Education": "#0F6E56", "Personal Care": "#5DCAA5",
}


# ── Auth helpers ─────────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed


def is_logged_in():
    return st.session_state.get("logged_in", False)


# ── Session state init ───────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "name" not in st.session_state:
    st.session_state.name = ""
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"   # "login" or "register"


# ════════════════════════════════════════════════════════════════
# LOGIN / REGISTER PAGE
# ════════════════════════════════════════════════════════════════
if not is_logged_in():
    st.markdown(
        "<h1 style='text-align:center;margin-top:3rem'>₹ AI Finance Advisor</h1>"
        "<p style='text-align:center;color:gray'>Your personal ML-powered budget assistant</p>",
        unsafe_allow_html=True
    )

    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        # Toggle between login and register
        tab_login, tab_register = st.tabs(["Login", "Create Account"])

        # ── LOGIN ──────────────────────────────────────────────
        with tab_login:
            st.markdown("### Welcome back")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_btn = st.form_submit_button("Login", use_container_width=True)

                if login_btn:
                    if not username or not password:
                        st.error("Please fill in both fields.")
                    else:
                        user = get_user(username.strip().lower())
                        if user and verify_password(password, user['password']):
                            st.session_state.logged_in = True
                            st.session_state.username  = user['username']
                            st.session_state.name      = user['name']
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

            # Demo account hint
            st.markdown(
                "<div style='font-size:12px;color:gray;margin-top:8px;text-align:center'>"
                "No account yet? Use the Create Account tab.</div>",
                unsafe_allow_html=True
            )

        # ── REGISTER ───────────────────────────────────────────
        with tab_register:
            st.markdown("### Create your account")
            with st.form("register_form"):
                reg_name     = st.text_input("Full Name")
                reg_username = st.text_input("Username")
                reg_password = st.text_input("Password", type="password")
                reg_confirm  = st.text_input("Confirm Password", type="password")
                reg_btn      = st.form_submit_button("Create Account", use_container_width=True)

                if reg_btn:
                    if not reg_name or not reg_username or not reg_password:
                        st.error("Please fill in all fields.")
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match.")
                    elif len(reg_password) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif get_user(reg_username.strip().lower()):
                        st.error("Username already exists. Choose another.")
                    else:
                        add_user(
                            reg_username.strip().lower(),
                            reg_name.strip(),
                            hash_password(reg_password)
                        )
                        st.success(f"Account created! You can now log in as '{reg_username}'.")

    st.stop()   # Don't render the rest of the app until logged in


# ════════════════════════════════════════════════════════════════
# MAIN APP — only shown after login
# ════════════════════════════════════════════════════════════════

# Seed data on first login
if os.path.exists(CSV_PATH):
    seed_from_csv(CSV_PATH)

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## ₹ AI Finance Advisor")
    st.markdown(f"👋 Hello, **{st.session_state.name}**")

    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.session_state.name      = ""
        st.rerun()

    st.markdown("---")
    st.markdown("### Add Expense")

    exp_desc = st.text_input("Description", placeholder="e.g. Swiggy order")
    predicted_category = "Food"
    if exp_desc.strip():
        result = predict_with_confidence(exp_desc.strip())
        predicted_category = result['category']
        st.markdown(
            f"<div style='background:#E1F5EE;border-radius:8px;padding:6px 10px;"
            f"font-size:12px;color:#085041;margin-bottom:8px'>"
            f"AI prediction: <b>{result['category']}</b> ({result['confidence']}% confident)</div>",
            unsafe_allow_html=True
        )

    with st.form("add_expense_form", clear_on_submit=True):
        exp_date    = st.date_input("Date", value=datetime.today())
        exp_amount  = st.number_input("Amount (₹)", min_value=1.0, step=10.0)
        default_idx = CATEGORIES.index(predicted_category) if predicted_category in CATEGORIES else 0
        exp_category = st.selectbox("Category (AI suggested)", CATEGORIES, index=default_idx)
        exp_payment  = st.selectbox("Payment Mode",
                                    ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"])
        submitted = st.form_submit_button("Add Transaction", use_container_width=True)
        if submitted:
            if exp_desc.strip() and exp_amount > 0:
                add_transaction(str(exp_date), exp_desc.strip(),
                                exp_amount, exp_category, exp_payment)
                st.success(f"Added: {exp_desc} → {exp_category}")
            else:
                st.error("Fill in description and amount.")

    st.markdown("---")
    st.markdown("### Filter by Month")
    now = datetime.today()
    sel_year  = st.selectbox("Year", list(range(2023, now.year + 1)), index=1)
    sel_month = st.selectbox("Month", list(range(1, 13)),
                              format_func=lambda m: datetime(2000, m, 1).strftime("%B"),
                              index=now.month - 1)

# ── Data ────────────────────────────────────────────────────────
df_month    = get_by_month(sel_year, sel_month)
df_all      = get_all_transactions()
month_label = datetime(sel_year, sel_month, 1).strftime("%B %Y")

# ── Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🔮 AI Prediction", "🚨 Anomaly Detection",
    "🤖 AI Advisor", "🧪 Try Classifier"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"## Dashboard — {month_label}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total = df_month["amount"].sum() if not df_month.empty else 0
        st.metric("Total Spent", f"₹{total:,.0f}")
    with c2:
        if not df_month.empty:
            top_cat = df_month.groupby("category")["amount"].sum().idxmax()
            top_amt = df_month.groupby("category")["amount"].sum().max()
            st.metric("Top Category", top_cat, f"₹{top_amt:,.0f}")
        else:
            st.metric("Top Category", "—")
    with c3:
        if not df_month.empty:
            biggest = df_month.loc[df_month["amount"].idxmax()]
            st.metric("Biggest Spend", f"₹{biggest['amount']:,.0f}", biggest["description"][:22])
        else:
            st.metric("Biggest Spend", "—")
    with c4:
        st.metric("Transactions", len(df_month) if not df_month.empty else 0)

    st.markdown("---")
    col_bar, col_pie = st.columns([3, 2])
    with col_bar:
        st.markdown("#### Spending by Category")
        if not df_month.empty:
            cat_df = df_month.groupby("category")["amount"].sum().reset_index().sort_values("amount")
            fig = px.bar(cat_df, x="amount", y="category", orientation="h",
                         color="category", color_discrete_map=CATEGORY_COLORS,
                         text=cat_df["amount"].apply(lambda x: f"₹{x:,.0f}"),
                         labels={"amount": "Amount (₹)", "category": ""})
            fig.update_layout(showlegend=False, height=340,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0, r=20, t=10, b=10))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No transactions this month.")

    with col_pie:
        st.markdown("#### Payment Modes")
        if not df_month.empty:
            pay_df = df_month.groupby("payment_mode")["amount"].sum().reset_index()
            fig2 = px.pie(pay_df, values="amount", names="payment_mode", hole=0.45,
                          color_discrete_sequence=["#1D9E75","#378ADD","#D85A30","#534AB7","#BA7517"])
            fig2.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation="h", yanchor="bottom", y=-0.25))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 6-Month Spending Trend")
    if not df_all.empty:
        df_all["month"] = pd.to_datetime(df_all["date"]).dt.to_period("M").astype(str)
        trend = df_all.groupby(["month","category"])["amount"].sum().reset_index()
        last6 = sorted(trend["month"].unique())[-6:]
        trend = trend[trend["month"].isin(last6)]
        fig3 = px.bar(trend, x="month", y="amount", color="category",
                      color_discrete_map=CATEGORY_COLORS, barmode="stack",
                      labels={"amount": "₹", "month": ""})
        fig3.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=10, b=10),
                           legend=dict(orientation="h", yanchor="bottom", y=-0.5))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Transactions table with DELETE ──────────────────────────
    st.markdown("---")
    st.markdown(f"#### Transactions — {month_label}")
    if not df_month.empty:
        st.markdown("Click 🗑️ to delete a transaction.")
        for _, row in df_month.iterrows():
            col_date, col_desc, col_amt, col_cat, col_pay, col_del = st.columns([1.5, 3, 1.5, 2, 1.5, 0.5])
            with col_date: st.write(row['date'])
            with col_desc: st.write(row['description'])
            with col_amt:  st.write(f"₹{row['amount']:,.0f}")
            with col_cat:  st.write(row['category'])
            with col_pay:  st.write(row['payment_mode'])
            with col_del:
                if st.button("🗑️", key=f"del_{row['id']}", help="Delete this transaction"):
                    delete_transaction(row['id'])
                    st.success("Deleted!")
                    st.rerun()
    else:
        st.info("No transactions found.")

# ════════════════════════════════════════════════════════════════
# TAB 2 — AI Prediction + SHAP
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## AI Expense Prediction")
    st.markdown("Linear Regression trained on 24 months of data predicts next month's total spend.")

    with st.spinner("Running prediction model..."):
        pred = predict_next_month()

    col_pred, col_range = st.columns(2)
    with col_pred:
        st.metric("Predicted spend next month", f"₹{pred['prediction']:,}")
    with col_range:
        st.metric("Expected range", f"₹{pred['low']:,} – ₹{pred['high']:,}")

    st.markdown("---")
    st.markdown("#### Actual spending vs predicted")
    history    = pred['history'].copy()
    last_month = pd.Period(history['month'].iloc[-1], 'M')
    next_month = str(last_month + 1)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Bar(x=history['month'], y=history['amount'],
                               name="Actual", marker_color="#378ADD"))
    fig_pred.add_trace(go.Bar(x=[next_month], y=[pred['prediction']],
                               name="Predicted", marker_color="#1D9E75",
                               error_y=dict(type='data',
                                            array=[pred['prediction'] - pred['low']],
                                            visible=True)))
    fig_pred.update_layout(height=320, barmode='group',
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=10, b=10),
                           legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                           yaxis=dict(title="Amount (₹)", gridcolor="rgba(128,128,128,0.15)"))
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Why this prediction? (SHAP Explainability)")
    st.markdown("Each bar shows how much a factor pushed the prediction **up (green)** or **down (red)**.")

    with st.spinner("Calculating SHAP values..."):
        explanation = get_shap_explanation()

    shap_df          = pd.DataFrame(explanation[:7])
    shap_df['label'] = shap_df['shap_value'].apply(lambda x: f"₹{abs(x):,}")
    colors_shap      = ["#1D9E75" if v > 0 else "#E24B4A" for v in shap_df['shap_value']]

    fig_shap = go.Figure(go.Bar(
        x=shap_df['shap_value'], y=shap_df['feature'],
        orientation='h', marker_color=colors_shap,
        text=shap_df['label'], textposition='outside'
    ))
    fig_shap.update_layout(height=320,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=60, t=10, b=10),
                           xaxis=dict(title="Impact on prediction (₹)",
                                      gridcolor="rgba(128,128,128,0.15)"),
                           yaxis=dict(title=""))
    st.plotly_chart(fig_shap, use_container_width=True)
    st.info("Green = pushes prediction higher | Red = pulls prediction lower | Length = impact size")

# ════════════════════════════════════════════════════════════════
# TAB 3 — Anomaly Detection
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Anomaly Detection")
    st.markdown("Flags transactions that are unusually high compared to your normal spending per category.")

    scope       = st.radio("Analyse", ["This month", "All transactions"], horizontal=True)
    df_to_check = df_month if scope == "This month" else df_all.drop(columns=['month'], errors='ignore')

    if df_to_check.empty:
        st.info("No transactions to analyse. Try 'All transactions'.")
    else:
        with st.spinner("Running anomaly detection..."):
            summary = get_anomaly_summary(df_to_check)

        a1, a2, a3 = st.columns(3)
        with a1: st.metric("Anomalies detected", summary['total_flagged'])
        with a2: st.metric("Anomalous amount", f"₹{summary['total_amount']:,.0f}")
        with a3:
            pct = (summary['total_flagged'] / len(df_to_check) * 100) if len(df_to_check) > 0 else 0
            st.metric("% of transactions", f"{pct:.1f}%")

        if summary['total_flagged'] == 0:
            st.success("No anomalies found. Your spending looks normal.")
        else:
            st.markdown("---")
            if not summary['by_category'].empty:
                st.markdown("#### Anomalies by Category")
                fig_anom = px.bar(summary['by_category'], x='category', y='count',
                                  color='category', color_discrete_map=CATEGORY_COLORS,
                                  text='count', labels={"count": "Flagged", "category": ""})
                fig_anom.update_layout(height=260, showlegend=False,
                                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                       margin=dict(l=0, r=0, t=10, b=10))
                fig_anom.update_traces(textposition='outside')
                st.plotly_chart(fig_anom, use_container_width=True)

            st.markdown("#### Flagged Transactions")
            flagged      = summary['flagged_df'].copy()
            flagged_disp = flagged[['date','description','amount','category','anomaly_reason']].copy()
            flagged_disp['amount']  = flagged_disp['amount'].apply(lambda x: f"₹{x:,.0f}")
            flagged_disp['z_score'] = flagged['z_score'].apply(lambda x: f"{x:.1f}σ")
            flagged_disp.columns    = ['Date','Description','Amount','Category','Why flagged','Z-score']
            st.dataframe(flagged_disp, use_container_width=True, hide_index=True)
            st.info("Z-score: 2σ = unusual, 5σ+ = very suspicious.")

# ════════════════════════════════════════════════════════════════
# TAB 4 — AI Advisor
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🤖 AI Budget Advisor")
    st.markdown(
        "Groq AI reads your ML model outputs — prediction, anomalies, spending pattern — "
        "and gives you 3 specific, actionable budget tips."
    )

    if df_month.empty:
        st.info("No transactions found for this month. Change the month filter in the sidebar.")
    else:
        if st.button("Generate AI Advice", type="primary"):
            with st.spinner("Analysing your finances with AI..."):
                pred_data  = predict_next_month()
                shap_data  = get_shap_explanation()
                anom_data  = get_anomaly_summary(df_month)
                summary    = build_summary_from_app(
                    df_month, pred_data, anom_data, shap_data, month_label
                )
                tips = get_ai_advice(summary)

            st.markdown("---")
            st.markdown(f"### Your personalised budget advice for {month_label}")

            tip_colors = ["#E1F5EE", "#EEF3FE", "#FFF8E1"]
            tip_icons  = ["💡", "📉", "🛡️"]
            for i, tip in enumerate(tips):
                color = tip_colors[i % len(tip_colors)]
                icon  = tip_icons[i % len(tip_icons)]
                st.markdown(
                    f"<div style='background:{color};border-radius:12px;"
                    f"padding:16px 20px;margin-bottom:12px'>"
                    f"<div style='font-size:15px;font-weight:600;margin-bottom:6px'>"
                    f"{icon} {tip['title']}</div>"
                    f"<div style='font-size:14px;color:#333;line-height:1.6'>"
                    f"{tip['advice']}</div></div>",
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown("#### Data used to generate this advice")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f"- **Month:** {month_label}")
                st.markdown(f"- **Total spent:** ₹{summary['total_spent']:,}")
                st.markdown(f"- **Top category:** {summary['top_category']} (₹{summary['top_category_amt']:,})")
                st.markdown(f"- **Predicted next month:** ₹{summary['predicted_next']:,}")
            with col_s2:
                st.markdown(f"- **Anomalies found:** {summary['anomaly_count']}")
                st.markdown(f"- **Anomalous amount:** ₹{summary['anomaly_amount']:,}")
                st.markdown(f"- **Key prediction driver:** {summary['shap_top_factor']}")

# ════════════════════════════════════════════════════════════════
# TAB 5 — Try the Classifier
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## Try the AI Classifier")
    st.markdown("Type any Indian merchant name and see the ML model predict its category instantly.")

    test_input = st.text_input("Test description", placeholder="e.g. Zomato, Ola, IRCTC, LIC premium")
    if test_input.strip():
        result = predict_with_confidence(test_input.strip())
        st.success(f"Predicted: **{result['category']}** — {result['confidence']}% confident")

        sorted_probs = sorted(result['all_probs'].items(), key=lambda x: x[1], reverse=True)
        cats  = [x[0] for x in sorted_probs]
        probs = [x[1] for x in sorted_probs]
        clrs  = ["#1D9E75" if c == result['category'] else "#B5D4F4" for c in cats]

        fig_cls = go.Figure(go.Bar(
            x=probs, y=cats, orientation='h', marker_color=clrs,
            text=[f"{p}%" for p in probs], textposition='outside'
        ))
        fig_cls.update_layout(height=360,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0, r=60, t=10, b=10),
                              xaxis=dict(title="Confidence (%)", range=[0, 120]),
                              yaxis=dict(title=""))
        st.plotly_chart(fig_cls, use_container_width=True)