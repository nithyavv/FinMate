"""
FinMate, Generative AI Financial Assistant — MVP

Run:
    streamlit run app.py

Features:
- Upload, simulate, or "connect" (simulated) to bank/UPI for transaction data (CSV-like)
- Auto-categorization of spending
- Simple insights (dining > X%, debt payments, subscriptions, etc.)
- RAG-style retrieval over:
    - Recent transactions
    - User-provided knowledge base docs
- Conversational Q&A with:
    - Tone adaptation (neutral / reassuring / encouraging)
    - Optional OpenAI LLM integration (if OPENAI_API_KEY set, legacy ChatCompletion style)
    - Fallback deterministic generator
- Explainability trace: shows retrieved snippets & insights used
- Elder-friendly mode (bigger text & simplified summary)
- Simulated Bank / UPI connector (Option 3 in data section)
- Net income metric (Income - Spend)
- Cashflow risk indicator (month-end risk)
- Recurring bills & subscriptions view
- Money Wellness Score (gamified scoring with explanation)
- Savings Goal Planner (goal amount + date → monthly saving)
"""

from __future__ import annotations
from datetime import datetime, timedelta, date
from typing import List, Dict, Any
import os
import re
import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv

try:
    import openai
except ImportError:
    openai = None

# ---------------------------
# Setup
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if openai is not None and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

st.set_page_config(
    page_title="FinMate, GenAI Financial Assistant",
    layout="wide"
)

# ---------------------------
# Data helpers
# ---------------------------

def sample_transactions() -> pd.DataFrame:
    """Generate a small synthetic transaction dataset for demo purposes."""
    now = datetime.now()
    rows = [
        {"timestamp": (now - timedelta(days=2)).isoformat(), "amount": -750.0, "currency": "INR", "merchant": "Cafe Aroma", "raw_desc": "CAFE AROMA ORDER", "account": "UPI-1111"},
        {"timestamp": (now - timedelta(days=4)).isoformat(), "amount": -850.0, "currency": "INR", "merchant": "Dominos", "raw_desc": "DOMINOS PIZZA", "account": "CARD-2222"},
        {"timestamp": (now - timedelta(days=12)).isoformat(), "amount": -12000.0, "currency": "INR", "merchant": "RentCo", "raw_desc": "MONTHLY RENT", "account": "NEFT-3333"},
        {"timestamp": (now - timedelta(days=1)).isoformat(), "amount": 50000.0, "currency": "INR", "merchant": "Employer Inc", "raw_desc": "SALARY CREDIT", "account": "NEFT-3333"},
        {"timestamp": (now - timedelta(days=30)).isoformat(), "amount": -1500.0, "currency": "INR", "merchant": "Electricity Board", "raw_desc": "ELECTRICITY BILL", "account": "UPI-1111"},
        {"timestamp": (now - timedelta(days=3)).isoformat(), "amount": -2000.0, "currency": "INR", "merchant": "Online Course", "raw_desc": "COURSE SUBSCRIPTION", "account": "CARD-2222"},
        {"timestamp": (now - timedelta(days=15)).isoformat(), "amount": -6000.0, "currency": "INR", "merchant": "Groceries Supermart", "raw_desc": "GROCERY STORE", "account": "CARD-2222"},
        {"timestamp": (now - timedelta(days=7)).isoformat(), "amount": -500.0, "currency": "INR", "merchant": "Coffee King", "raw_desc": "COFFEE", "account": "UPI-1111"},
        {"timestamp": (now - timedelta(days=60)).isoformat(), "amount": -4000.0, "currency": "INR", "merchant": "Loan EMIs", "raw_desc": "AUTO LOAN EMI", "account": "NEFT-3333"},
    ]
    return pd.DataFrame(rows)


CATEGORY_RULES = [
    (r"rent|apartment|housing", "Housing"),
    (r"salary|credit", "Income"),
    (r"dominos|pizza|restaurant|order|dine|cafe|coffee", "Dining"),
    (r"grocery|supermart|grocer|market", "Groceries"),
    (r"electric|bill|utility", "Utilities"),
    (r"loan|emi|mortgage", "Debt"),
    (r"course|subscription|netflix|prime", "Subscriptions"),
    (r".*", "Other"),
]


def categorize(text: str) -> str:
    """Apply simple regex-based category rules to a description string."""
    if not isinstance(text, str):
        return "Other"
    t = text.lower()
    for pattern, cat in CATEGORY_RULES:
        if re.search(pattern, t):
            return cat
    return "Other"


def detect_tone(user_input: str) -> str:
    """Very simple sentiment/tone detection."""
    low = (user_input or "").lower()
    if any(w in low for w in ["worried", "scared", "stressed", "anxious", "concerned", "panic"]):
        return "reassuring"
    if any(w in low for w in ["confident", "ok", "fine", "good", "great", "sure"]):
        return "encouraging"
    return "neutral"


class SimpleRetriever:
    """Tiny TF-IDF based retriever for RAG-style context."""

    def __init__(self) -> None:
        self.docs: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self.vec = None
        self.vectorizer: TfidfVectorizer | None = None

    def add_docs(self, texts: List[str], metas: List[Dict[str, Any]]) -> None:
        """Add docs and build TF-IDF matrix."""
        if not texts:
            return
        self.docs.extend(texts)
        self.meta.extend(metas)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self.vec = self.vectorizer.fit_transform(self.docs)

    def query(self, q: str, topk: int = 5) -> List[Dict[str, Any]]:
        """Return top-k most similar documents."""
        if not self.docs or self.vectorizer is None or self.vec is None:
            return []
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.vec).flatten()
        top_idx = sims.argsort()[::-1][:topk]
        results = []
        for i in top_idx:
            results.append(
                {
                    "score": float(sims[i]),
                    "text": self.docs[i],
                    "meta": self.meta[i],
                }
            )
        return results


def deterministic_generator(context: Dict[str, Any], question: str, tone: str) -> str:
    """
    Simple fallback "AI" when no LLM is available.

    It uses:
    - high_spend_categories
    - upcoming_bills
    - heuristic insights (if provided in context)
    """
    lines: List[str] = []
    lines.append(f"Question: {question}")
    if tone == "reassuring":
        intro = "I hear that you’re concerned. Let’s look at your finances calmly and step by step."
    elif tone == "encouraging":
        intro = "Nice that you’re proactively thinking about this. Here’s a plan you can try."
    else:
        intro = "Here’s what your recent financial pattern suggests."

    lines.append(intro)
    lines.append("")

    high_spend = context.get("high_spend_categories", "")
    if high_spend:
        lines.append(f"- Your main spending areas recently are: **{high_spend}**.")

    upcoming_bills = context.get("upcoming_bills", [])
    if upcoming_bills:
        lines.append("- Upcoming bills to watch:")
        for b in upcoming_bills:
            lines.append(f"  • {b}")

    insights = context.get("insights", [])
    if insights:
        lines.append("")
        lines.append("Key observations from your data:")
        for ins in insights:
            lines.append(f"- ({ins.get('severity','info')}) {ins.get('title')}: {ins.get('explain')}")

    lines.append("")
    lines.append("Practical next steps:")
    lines.append("- Try to set aside 5–10% of your latest salary into a dedicated savings/goal bucket.")
    lines.append("- Pick one high-spend category and set a small cutback target for next month (e.g., 10–15%).")
    lines.append("- Review any subscriptions and debt payments to see if anything can be optimized or reprioritized.")

    lines.append("")
    lines.append("Explainability:")
    lines.append("- This suggestion is based on your last ~30 days of transactions and detected categories.")
    lines.append("- I focused on high-spend categories, recurring-like items, and debt-related payments.")

    return "\n".join(lines)


def call_llm_with_openai(prompt: str) -> str:
    """
    Optional: call OpenAI ChatCompletion API (legacy style).
    Uses gpt-4o-mini for cost-effective reasoning.
    """
    if openai is None or not OPENAI_API_KEY:
        raise RuntimeError("OpenAI not configured.")

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a concise, empathetic personal financial assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.25,
        max_tokens=450,
    )
    return resp["choices"][0]["message"]["content"]


# ---------------------------
# Helper: Money wellness score
# ---------------------------

def compute_wellness_score(
    total_income: float,
    total_spend: float,
    dining_share: float,
    has_debt: bool,
) -> Dict[str, Any]:
    """
    Very simple gamified score out of 100.

    - Base score: 50
    - + up to 30 points for savings rate
    - - up to 10 points for high dining share
    - - up to 10 points if debt present
    """
    explanations: List[str] = []
    score = 50

    savings = max(total_income - total_spend, 0)
    savings_rate = (savings / total_income) if total_income > 0 else 0

    # Savings contribution
    if savings_rate >= 0.3:
        score += 30
        explanations.append("High savings rate (≥30% of income) adds +30 points.")
    elif savings_rate >= 0.15:
        score += 20
        explanations.append("Decent savings rate (15–30% of income) adds +20 points.")
    elif savings_rate >= 0.05:
        score += 10
        explanations.append("Some savings (5–15% of income) adds +10 points.")
    else:
        explanations.append("Low savings rate (<5% of income) adds no extra points.")

    # Dining penalty
    if dining_share > 0.3:
        score -= 10
        explanations.append("Very high dining share (>30% of spend) subtracts 10 points.")
    elif dining_share > 0.2:
        score -= 5
        explanations.append("Somewhat high dining share (20–30% of spend) subtracts 5 points.")
    else:
        explanations.append("Dining spend is within a reasonable range.")

    # Debt penalty
    if has_debt:
        score -= 10
        explanations.append("Debt/EMI payments detected → -10 points for risk.")
    else:
        explanations.append("No debt payments detected in recent data.")

    score = max(0, min(100, int(round(score))))
    return {"score": score, "details": explanations, "savings_rate": savings_rate}


# ---------------------------
# Streamlit UI Layout
# ---------------------------

st.title("💰 FinMate, Generative AI Financial Assistant")
st.caption("Personalized, conversational, explainable money guidance (demo).")

# Accessibility toggle (elder-friendly mode)
elder_mode = st.sidebar.checkbox("Elder-friendly mode (larger text & simplified view)", value=False)

# Simple CSS for elder-friendly mode
if elder_mode:
    st.markdown(
        """
        <style>
        body, .stMarkdown, .stTextInput, .stTextArea, .stMetric, .stDataFrame {
            font-size: 18px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("About this demo")
    st.write(
        """
        - Upload your own CSV or use sample data.
        - Or simulate a Bank / UPI connection.
        - Data is processed locally in this demo.
        - Optional: add knowledge snippets (goals, rules).
        - Ask questions in natural language.
        """
    )
    if st.button("Reset session"):
        st.session_state.clear()
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("LLM Status")
    if OPENAI_API_KEY and openai is not None:
        st.success("OpenAI key detected – LLM mode enabled.")
    else:
        st.warning("No OpenAI key – using deterministic offline responses.")


# Initialize session state
if "tx" not in st.session_state:
    st.session_state["tx"] = sample_transactions()

if "kb_docs" not in st.session_state:
    st.session_state["kb_docs"] = []  # each: {title, text}

if "data_source" not in st.session_state:
    st.session_state["data_source"] = "sample"  # 'sample' | 'csv' | 'bank'

# ---------------------------
# 1) Data Upload & Preview / Bank Connect
# ---------------------------
st.header("Connect Accounts or Upload Data")

upload_col, sample_col, bank_col = st.columns([2, 1, 1])

with upload_col:
    st.markdown("**Upload bank statement (CSV)**")
    uploaded = st.file_uploader(
        "Upload CSV with at least: timestamp, amount, merchant, raw_desc (account optional).",
        type=["csv"],
    )
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["tx"] = df
            st.session_state["data_source"] = "csv"
            st.success("Transactions loaded from uploaded CSV (simulated bank statement).")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

with sample_col:
    st.markdown("**Use sample dataset (Demo Purpose)**")
    if st.button("Use sample dataset"):
        st.session_state["tx"] = sample_transactions()
        st.session_state["data_source"] = "sample"
        st.info("Sample transactions loaded.")

with bank_col:
    st.markdown("**Connect to Bank / UPI (Simulated)**")
    st.write(
        "In production, this would use Account Aggregator / bank APIs "
        "with user consent to fetch transactions securely."
    )
    if st.button("Connect bank (demo)"):
        st.session_state["tx"] = sample_transactions()
        st.session_state["data_source"] = "bank"
        st.success("Bank connection simulated. Recent transactions fetched (demo).")

source_label = {
    "sample": "Sample dataset",
    "csv": "Uploaded CSV (simulated bank statement)",
    "bank": "Simulated live bank / UPI connection",
}.get(st.session_state.get("data_source", "sample"), "Sample dataset")

st.caption(f"Current transaction source: **{source_label}**")

tx = st.session_state["tx"].copy()

# Normalize / clean
if "timestamp" in tx.columns:
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")
else:
    st.error("Missing 'timestamp' column in data.")
if "amount" in tx.columns:
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
else:
    st.error("Missing 'amount' column in data.")

if "merchant" not in tx.columns:
    tx["merchant"] = tx.get("raw_desc", "Unknown")

tx["raw_desc"] = tx.get("raw_desc", tx["merchant"]).fillna("Unknown")
tx["merchant"] = tx["merchant"].fillna("Unknown")

tx["category"] = tx["raw_desc"].apply(categorize)

st.subheader("Transactions Preview")
st.dataframe(
    tx.sort_values("timestamp", ascending=False).reset_index(drop=True),
    use_container_width=True,
)

# ---------------------------
# 2) Analytics & Insights
# ---------------------------
st.header("Quick Analytics & Insights")

last_30_mask = tx["timestamp"] >= (pd.Timestamp.now() - pd.Timedelta(days=30))
last_30 = tx[last_30_mask].copy()

total_income = last_30[last_30["amount"] > 0]["amount"].sum()
total_spend = -last_30[last_30["amount"] < 0]["amount"].sum()

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Income (last 30 days)", f"{total_income:,.2f} INR")
with m2:
    st.metric("Spend (last 30 days)", f"{total_spend:,.2f} INR")
with m3:
    net = total_income - total_spend
    st.metric("Net (Income - Spend)", f"{net:,.2f} INR")

st.write("Top spend categories (last 30 days):")

# Aggregate by category, consider absolute spend
cat_summary = (
    last_30.assign(abs_amount=lambda d: d["amount"].apply(lambda x: -x if x < 0 else x))
    .groupby("category")["abs_amount"]
    .sum()
    .sort_values(ascending=False)
)
st.table(cat_summary.reset_index().head(10))

# Heuristic insight rules
insights: List[Dict[str, Any]] = []

if not cat_summary.empty:
    total_abs = cat_summary.sum()
    dining_val = cat_summary.get("Dining", 0.0)
    dining_share = float(dining_val / total_abs) if total_abs > 0 else 0.0
    if dining_val > 0.2 * total_abs:
        insights.append(
            {
                "id": "ins1",
                "severity": "medium",
                "title": "High dining spend",
                "explain": f"Dining is about {dining_val:,.0f} INR, more than 20% of your total spend in the last 30 days.",
            }
        )
else:
    dining_share = 0.0

subs = tx[tx["category"] == "Subscriptions"]
if len(subs) > 0:
    insights.append(
        {
            "id": "ins2",
            "severity": "low",
            "title": "Subscriptions detected",
            "explain": f"{len(subs)} subscription-like transactions detected. You may want to review them.",
        }
    )

debt = tx[tx["category"] == "Debt"]
has_debt = len(debt) > 0
if has_debt:
    insights.append(
        {
            "id": "ins3",
            "severity": "high",
            "title": "Debt payments present",
            "explain": "Recent transactions show debt/EMI payments. Consider reviewing interest rates and payoff strategy.",
        }
    )

if not insights:
    insights.append(
        {
            "id": "ins0",
            "severity": "info",
            "title": "No obvious risks",
            "explain": "No strong risk patterns detected in this small dataset.",
        }
    )

st.subheader("Insights")
for ins in insights:
    st.markdown(f"**➤ {ins['title']}** _({ins['severity']})_")
    st.write(ins["explain"])

# ---------------------------
# 2a) Extended Analysis: Cashflow, Recurring, Wellness
# ---------------------------
st.markdown("### Extended Analysis")

col_a, col_b, col_c = st.columns(3)

# Cashflow risk indicator
with col_a:
    days_left = 30  # simple assumption for demo
    avg_daily_spend = (total_spend / 30.0) if total_spend > 0 else 0
    projected_spend = avg_daily_spend * days_left
    projected_balance = total_income - projected_spend
    if projected_balance < 0:
        risk_msg = "High risk of running short by month-end."
    elif projected_balance < 0.1 * total_income:
        risk_msg = "Tight but manageable. Consider reducing variable spend."
    else:
        risk_msg = "Cashflow looks comfortable for this month."
    st.markdown("**Cashflow Outlook**")
    st.write(f"Projected month-end balance (rough): {projected_balance:,.0f} INR")
    st.write(risk_msg)

# Recurring bills & subscriptions
with col_b:
    st.markdown("**Recurring Bills & Subscriptions**")
    recurring = tx[tx["category"].isin(["Housing", "Utilities", "Debt", "Subscriptions"])]
    if recurring.empty:
        st.write("No obvious recurring bills detected in this sample.")
    else:
        show_cols = ["timestamp", "merchant", "amount", "category", "raw_desc"]
        st.dataframe(
            recurring.sort_values("timestamp", ascending=False)[show_cols].head(10),
            height=200,
        )

# Money wellness score
with col_c:
    st.markdown("**Money Wellness Score**")
    score_info = compute_wellness_score(total_income, total_spend, dining_share, has_debt)
    st.metric("Wellness Score (0–100)", score_info["score"])
    with st.expander("Why this score?"):
        for line in score_info["details"]:
            st.write("- " + line)

# Elder-friendly simple summary
if elder_mode:
    st.markdown("---")
    st.markdown("### 👵 Elder-friendly Summary")
    st.write(
        f"In the last 30 days, income was about **{total_income:,.0f} INR** and spending was about "
        f"**{total_spend:,.0f} INR**. Your money wellness score is **{score_info['score']} / 100**."
    )
    st.write("Focus on: keeping some savings, watching dining and subscriptions, and managing any EMIs calmly.")

st.markdown("---")

# ---------------------------
# 3) Knowledge Base (for RAG)
# ---------------------------
st.header("Knowledge Base (Goals, Rules, Preferences)")

kb_col1, kb_col2 = st.columns([2, 1])

with kb_col1:
    kb_title = st.text_input("KB Title", placeholder="Example: My savings rule")
    kb_text = st.text_area(
        "KB Text",
        placeholder="Example: I want to save at least 10% of my salary for travel each month.",
    )
    if st.button("Add KB Document"):
        if kb_text.strip():
            st.session_state["kb_docs"].append(
                {
                    "title": kb_title or f"doc_{len(st.session_state['kb_docs']) + 1}",
                    "text": kb_text.strip(),
                }
            )
            st.success("Knowledge document added.")
        else:
            st.warning("Please enter some text for the KB document.")

with kb_col2:
    st.write("Existing KB documents:")
    if not st.session_state["kb_docs"]:
        st.write("No KB docs yet.")
    else:
        for i, d in enumerate(st.session_state["kb_docs"], start=1):
            st.write(f"{i}. **{d['title']}** – {d['text'][:80]}...")

st.markdown("---")

# ---------------------------
# 3a) Savings Goal Planner
# ---------------------------
st.header("Savings Goal Planner")

goal_col1, goal_col2, goal_col3 = st.columns(3)
with goal_col1:
    goal_amount = st.number_input("Goal amount (INR)", min_value=0.0, value=100000.0, step=1000.0)
with goal_col2:
    goal_date = st.date_input("Target date", value=date.today())
with goal_col3:
    current_savings = st.number_input("Current savings for this goal (optional)", min_value=0.0, value=0.0, step=1000.0)

if st.button("Calculate Goal Plan"):
    days_to_goal = (goal_date - date.today()).days
    if days_to_goal <= 0:
        st.error("Please choose a future date for your goal.")
    else:
        remaining = max(goal_amount - current_savings, 0)
        months_to_goal = max(days_to_goal / 30.0, 1)
        required_per_month = remaining / months_to_goal
        st.success(
            f"To reach **{goal_amount:,.0f} INR** by **{goal_date}**, "
            f"you need to save approximately **{required_per_month:,.0f} INR per month**."
        )
        if total_income > 0:
            share = required_per_month / total_income
            st.write(f"That is about **{share*100:.1f}%** of your recent monthly income.")

st.markdown("---")

# ---------------------------
# 4) Conversational Assistant (RAG-style)
# ---------------------------
st.header("Ask the Assistant")

question = st.text_area(
    "Ask anything about your finances:",
    placeholder="Examples: Can I afford a vacation next month? How to reduce my loan burden?",
    height=120,
)

auto_tone = detect_tone(question)
st.write(f"Detected tone based on your message: **{auto_tone}**")

if st.button("Ask Assistant"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # Build retriever
        retriever = SimpleRetriever()

        docs: List[str] = []
        metas: List[Dict[str, Any]] = []

        # Add KB docs
        for d in st.session_state["kb_docs"]:
            docs.append(d["text"])
            metas.append({"source": "kb", "title": d["title"]})

        # Add recent transaction snippets
        tx_sorted = tx.sort_values("timestamp", ascending=False).head(40)
        for idx, row in tx_sorted.iterrows():
            snippet = (
                f"{row['timestamp'].date()} | {row['merchant']} | {row['amount']} | "
                f"{row['category']} | {row['raw_desc']}"
            )
            docs.append(snippet)
            metas.append({"source": "transaction", "tx_index": int(idx)})

        retriever.add_docs(docs, metas)

        # Query retriever
        retrieved = retriever.query(question, topk=5)

        st.subheader("Context used by Assistant")
        if not retrieved:
            st.write("No context retrieved (empty documents).")
        else:
            for r in retrieved:
                src = r["meta"].get("source", "unknown")
                st.write(f"- (score {r['score']:.3f}) [{src}] {r['text'][:200]}")

        # High-level context
        high_spend_categories = ", ".join(cat_summary.index[:3]) if not cat_summary.empty else ""
        context_payload: Dict[str, Any] = {
            "high_spend_categories": high_spend_categories,
            "upcoming_bills": [],  # placeholder for future logic
            "insights": insights,
            "retrieved_texts": [r["text"] for r in retrieved],
        }

        # Build prompt for LLM (if used)
        answer_text: str
        if OPENAI_API_KEY and openai is not None:
            st.info("Using OpenAI LLM for the answer...")
            prompt = (
                "You are a friendly, explainable financial assistant. "
                "Use the following context (transactions + user rules) to answer briefly:\n\n"
            )
            prompt += f"High-spend categories: {high_spend_categories}\n"
            prompt += "Heuristic insights:\n"
            for ins in insights:
                prompt += f"- {ins['title']}: {ins['explain']}\n"
            prompt += "\nRetrieved snippets:\n"
            for r in retrieved:
                prompt += f"- {r['text']}\n"
            prompt += "\nUser question:\n"
            prompt += question
            prompt += (
                "\n\nRespond with:\n"
                "- A short, empathetic answer\n"
                "- 2–3 practical steps\n"
                "- A short rationale mentioning what data you used\n"
            )

            try:
                answer_text = call_llm_with_openai(prompt)
            except Exception as e:
                st.error(f"OpenAI call failed, falling back to deterministic generator. Error: {e}")
                answer_text = deterministic_generator(context_payload, question, auto_tone)
        else:
            st.info("No LLM configured – using deterministic offline reasoning.")
            answer_text = deterministic_generator(context_payload, question, auto_tone)

        st.subheader("Assistant Answer")
        st.write(answer_text)

        # Explainability trace
        st.subheader("Explainability Trace (Debug View)")
        trace = {
            "retrieved_count": len(retrieved),
            "retrieved_examples": [r["text"][:200] for r in retrieved],
            "high_spend_categories": high_spend_categories,
            "insights_used": insights,
            "wellness_score": score_info,
            "data_source": st.session_state.get("data_source", "sample"),
        }
        st.json(trace)

# ---------------------------
# 5) Export
# ---------------------------
st.markdown("---")
st.header("Export Data")

csv_export = tx.to_csv(index=False)
st.download_button(
    "Download current transaction data as CSV",
    data=csv_export,
    file_name="transactions_export.csv",
    mime="text/csv",
)
st.caption("You can reuse this CSV in further experiments or as sample data for the judges.")
