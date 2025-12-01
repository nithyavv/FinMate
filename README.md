# FinMate - Generative AI Financial Assistant ( MVP)

**Personalized • Explainable • Conversational • Financial Wellness**

FinMate is a **Generative AI–powered personal finance assistant** designed to make money management **simple, contextual, and actionable** for everyone - especially young professionals, families, freelancers, and the elderly.

This MVP demonstrates how AI + lightweight analytics + RAG can turn raw transaction data into **clear insights**, **human-like conversations**, and **smart financial guidance**.

---

## Features Included in This MVP

### 1. Transaction Upload / Simulated Bank Connect
- Upload CSV bank statements  
- Use sample dataset  
- Simulate a “Bank / UPI connection” (demo mode)

### 2. Automatic Categorization
Uses rule-based classifier (Dining, Debt, Groceries, Subscriptions, Income, Utilities, Other).

### 3. Quick Analytics
- Income vs Spend (last 30 days)  
- Net Income (Income – Spend)  
- Category-wise spend summary  
- Subscription detection  
- Debt/EMI detection  
- Dining overspend alerts  

### 4. Enhanced Analytics
- **Cashflow Outlook** (month-end risk forecast)  
- **Recurring Bills Dashboard**  
- **Money Wellness Score (0-100)** with breakdown  
- **Savings Rate**  
- **Lifestyle Insight Flags**  

### 5. Savings Goal Planner
Enter:
- Goal amount  
- Target date  
- Current savings  

→ App calculates required monthly saving.

### 6. Elder-Friendly Mode
- Larger text  
- Simplified summary  
- Clear tone and instructions  

### 7. Conversational Assistant (RAG-style)
- TF-IDF retriever over:
  - Recent transactions  
  - User’s knowledge base docs  
- Tone adaptation (reassuring, encouraging, neutral)  
- Uses Generative AI (OpenAI) if available  
- Falls back to rule-based reasoning if no API key  

### 8. Explainability Panel
Shows what data the assistant used:
- Retrieved contexts  
- High-spend categories  
- Insights  
- Used documents/rules  

---

## Tech Stack (Hackathon Version)

### User Interface
- Streamlit

### Core Logic / Data Processing
- Python  
- Pandas  
- NumPy  

### AI / ML
- Scikit-learn (TF-IDF + cosine similarity retriever)  
- OpenAI ChatCompletion API (optional, legacy client `openai==0.28`)  
- Deterministic fallback generator  

### Config & Secrets
- python-dotenv  

### Storage
- Streamlit session state (in-memory)

**No database or vector DB required** for this MVP → lightweight and portable.

---

## Folder Structure

finmate/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
└── sample_data/
    └── sample_transactions.csv

