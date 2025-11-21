# Generative AI Financial Assistant

## 1. Problem

Traditional banking apps show numbers, not guidance.  
Young professionals, freelancers, and families (especially in emerging markets) struggle with:

- Understanding spending patterns
- Managing debt and credit
- Planning realistic savings and goals

We ask: **What if your money app talked like a trusted advisor?**

---

## 2. Solution

A **Generative AI–powered financial assistant** that:

- Connects to transaction data (CSV in this MVP)
- Detects patterns in spending, income, and debt
- Uses a RAG-style approach to answer natural language questions
- Speaks in a friendly, contextual tone based on user sentiment
- Provides **explainable** insights (which transactions influenced the advice)

### Core Features in This MVP

- Upload or simulate bank transactions
- Auto-categorize spending (Dining, Groceries, Debt, etc.)
- Show basic analytics: income vs spend, top categories
- Heuristic **“insights”** (e.g., dining spend is high, debt payments detected)
- User can add personal **knowledge snippets** (budgets, rules, goals)
- Ask natural questions:
  - “Can I afford a vacation next month?”
  - “How do I reduce my loan burden?”
- Retrieval over:
  - Recent transactions
  - User KB docs
- Explainability:
  - Shows which text snippets were retrieved
  - Lists heuristic insights used by the assistant
- Tone adapts based on user message (neutral / reassuring / encouraging)

---

## 3. Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **Core Libraries:**
  - `pandas`, `numpy` — data handling
  - `scikit-learn` — simple TF-IDF retrieval
  - `streamlit` — UI
- **Optional GenAI:**
  - `openai` (only if `OPENAI_API_KEY` is set)

This MVP is intentionally **single-service and single-file** for hackathon.

---

## 4. Architecture (MVP View)

**Data Layer:**

- CSV-based transactions or sample data
- Derived attributes: categories, simple insights
- “Knowledge base” snippets stored in memory

**AI/ML:**

- Simple rules for category detection (regex-based)
- TF-IDF-based retriever over:
  - KB documents
  - Recent transaction snippets
- Optional LLM (OpenAI) to generate natural language answers
- Deterministic “fallback” generator if no LLM key is provided

**Explainability:**

- Shows retrieved snippets & heuristic insights
- Short rationale-style answer from the assistant

---

## 5. How to Run

streamlit run app.py
