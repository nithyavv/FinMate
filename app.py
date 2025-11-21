"""
FinMate, Generative AI Financial Assistant —  MVP

Run:
    streamlit run app.py

Features:
- Upload or simulate transaction data (CSV)
- Auto-categorization of spending
- Simple insights (dining > X%, debt payments, subscriptions, etc.)
- RAG-style retrieval over:
    - Recent transactions
    - User-provided knowledge base docs
- Conversational Q&A with:
    - Tone adaptation (neutral / reassuring / encouraging)
    - Optional OpenAI LLM integration (if OPENAI_API_KEY set)
    - Fallback deterministic generator
- Explainability trace: shows retrieved snippets & insights used
"""

from __future__ import annotations
from datetime import datetime, timedelta
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

# Optional: OpenAI
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
    Optional: call OpenAI ChatCompletion API.
    Uses gpt-4o-mini for cost-effective reasoning.
    """
    if openai is None or not OPENAI_API_KEY:
        raise RuntimeError("OpenAI not configured.")

    # NOTE: This uses the classic ChatCompletion style for hackathon simplicity.
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
# Streamlit UI Layout
# ---------------------------

st.title("💰 FinMate, Generative AI Financial Assistant —  MVP")
st.caption("Personalized, conversational, explainable money guidance (demo).")

with st.sidebar:
    st.header("About this demo")
    st.write(
        """
        - Upload your own CSV or use sample data.
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

# ---------------------------
# 1) Data Upload & Preview
# ---------------------------
st.header("1. Upload Transactions or Use Sample")

upload_col, sample_col = st.columns([2, 1])

with upload_col:
    uploaded = st.file_uploader(
        "Upload CSV with at least: timestamp, amount, merchant, raw_desc (account optional).",
        type=["csv"],
    )
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["tx"] = df
            st.success("Transactions loaded from CSV.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

with sample_col:
    if st.button("Use sample dataset"):
        st.session_state["tx"] = sample_transactions()
        st.info("Sample transactions loaded.")

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
st.header("2. Quick Analytics & Insights")

last_30_mask = tx["timestamp"] >= (pd.Timestamp.now() - pd.Timedelta(days=30))
last_30 = tx[last_30_mask].copy()

total_income = last_30[last_30["amount"] > 0]["amount"].sum()
total_spend = -last_30[last_30["amount"] < 0]["amount"].sum()

m1, m2 = st.columns(2)
with m1:
    st.metric("Income (last 30 days)", f"{total_income:,.2f} INR")
with m2:
    st.metric("Spend (last 30 days)", f"{total_spend:,.2f} INR")

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
    if dining_val > 0.2 * total_abs:
        insights.append(
            {
                "id": "ins1",
                "severity": "medium",
                "title": "High dining spend",
                "explain": f"Dining is about {dining_val:,.0f} INR, more than 20% of your total spend in the last 30 days.",
            }
        )

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
if len(debt) > 0:
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

st.markdown("---")

# ---------------------------
# 3) Knowledge Base (for RAG)
# ---------------------------
st.header("3. Knowledge Base (Goals, Rules, Preferences)")

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
# 4) Conversational Assistant (RAG-style)
# ---------------------------
st.header("4. Ask the Assistant")

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
        }
        st.json(trace)

# ---------------------------
# 5) Export
# ---------------------------
st.markdown("---")
st.header("5. Export Data")

csv_export = tx.to_csv(index=False)
st.download_button(
    "Download current transaction data as CSV",
    data=csv_export,
    file_name="transactions_export.csv",
    mime="text/csv",
)
st.caption("You can reuse this CSV in further experiments or as sample data for the judges.")
