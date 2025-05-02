# 🧠 LLM Engineer Technical Assessment: Financial Trading Intelligence

## 📍 Overview

This technical assessment evaluates your ability to build a hybrid AI system that leverages LLMs for financial decision-making in algorithmic trading. Using a structured crypto dataset (`gold.csv`), you will implement a trading signal generator, evaluate its reasoning and performance, and improve it through experimentation.

You are expected to demonstrate hands-on skills in LLM integration, prompt engineering, reasoning analysis, and hybrid modeling with structured data — all applied to a real-world financial setting.

---

## 📊 Dataset Description

The dataset `gold.csv` includes both structured market data and unstructured sentiment fields:

| Column                  | Description                                              |
|-------------------------|----------------------------------------------------------|
| `datetime`              | Timestamp                                                |
| `close`                 | BTC closing price                                        |
| `mvrv_btc_momentum`     | MVRV momentum indicator                                  |
| `spot_volume_daily_sum`| Daily trading volume                                      |
| `Signal`                | Ground truth signal (-1 or 1)                            |
| `summary`               | News summary at that time                                |
| `next_news_prediction`  | Predicted news narrative                                 |
| `sentiment`             | Market sentiment label (Fear, Greed, etc.)               |
| `index`                 | Sentiment index score (numeric)                          |
| `key_factors`           | Drivers behind market movements                          |
| `dominant_emotions`     | Emotional context of the market                          |
| `dominant_sentiment`    | Main sentiment class                                     |
| `intensity`             | Sentiment intensity score (1–10)                         |
| `psychology_explanation`| Narrative explaining market psychology                   |

---
We have also added an additional table (`gold_full_table.csv`) if you want to work with a larger dataset.

## ✅ Tasks

### 🧩 Task 1: Baseline Trading System

**Objective:**  
Build an LLM-powered system that outputs a continuous trading decision ∈ (-1, 1) for each row in the dataset. Your system should apply some kind of RAG. Be creative, rags can be used in many ways not only for documents.
#### Signal Interpretation

- **1 (Buy/Long)**: When a buy signal is generated, you take a long position. You'll profit if the price increases after your entry.
- **-1 (Sell/Short)**: When a sell signal is generated, you take a short position. You'll profit if the price decreases after your entry.
- **0 (Neutral)**: No position is taken. Market price movements have no impact on your portfolio during this period.

**Requirements:**
- Use LLMs to reason over sentiment and psychology fields.
- Combine structured and unstructured data to enhance output quality (hybrid system encouraged). With price you can build multiple technical indicators for example.
- Avoid look-ahead bias and overfitting — we will evaluate your system on a future data period.
- Be creative: showcase your full skillset across LLMs, embeddings, retrieval, and hybrid ML techniques.

**Deliverables (`/task_1_baseline/`):**
- `baseline_strategy.ipynb` notebook:
  - Loads and prepares the dataset
  - Defines your LLM/hybrid model pipeline
  - Outputs a new column with decision values ∈ (-1, 1)
  - Explains your modeling approach and assumptions

---

### 📊 Task 2: Evaluation Framework

**Objective:**  
Use the provided `Backtester` class to simulate trading performance **and** create an additional evaluation module that inspects the quality of model reasoning and decisions.

**Requirements:**
- Run the provided backtesting tool to generate financial metrics and a PnL curve.
- Create an **LLM/system reasoning evaluation module** that:
  - Analyzes a sample of outputs and associated explanations or retrieved context
  - Identifies decision inconsistencies, hallucinations, or overconfidence
  - Provides insights into where and why the model tends to fail or succeed
  - Proposes at least two improvement vectors to guide Task 3

**Deliverables (`/task_2_evaluation/`):**
- `evaluation_report.ipynb` notebook:
  - Shows backtester output and visual performance plots
  - Includes a dedicated section for **qualitative and/or statistical analysis** of model reasoning
  - Proposes concrete areas for improvement with supporting evidence

---

### ⚙️ Task 3: Experimentation & Optimization

**Objective:**  
Select two improvement directions based on Task 2 insights, implement them, and measure the impact.

**Examples:**
- Better prompt design or template selection
- Embedding-based memory or retrieval improvements
- Confidence scoring and output filtering
- Alternate LLM providers or fine-tuned models
- Custom aggregation of structured + LLM outputs

**Deliverables (`/task_3_experimentation/`):**
- `experiments_and_results.ipynb` notebook:
  - Documents your two improvements
  - Runs comparative tests and analyzes the new performance
  - Summarizes what improved, what didn’t, and why

---

## 📂 Submission Instructions

Structure your GitHub repo as follows:

```
repo/
│
├── task_1_baseline/
│   └── baseline_strategy.ipynb
│
├── task_2_evaluation/
│   └── evaluation_report.ipynb
│
├── task_3_experimentation/
│   └── experiments_and_results.ipynb
│
├── gold.csv           # Dataset file
├── backtester.py      # Provided backtesting module
└── README.md          #
```

## 📈 Evaluation Criteria

| Area                              | What We’re Looking For                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------------------|
| **Code Quality**                  | Modular, well-commented, reproducible notebooks                                        |
| **LLM System Design**             | Effective use of LLMs (with or without hybrid elements) for market reasoning            |
| **Backtest Performance**          | Reasonable returns and stability — no unrealistic overfitting                          |
| **LLM Reasoning Evaluation**      | Insightful analysis of model decisions, with evidence-based improvement proposals      |
| **Experimentation Rigor**         | Structured comparison of system improvements                                           |
| **Communication & Clarity**      | Notebooks clearly explain rationale, logic, and results                                |

---

## 🌟 Bonus Points

- Integration with vector databases or multiple retrieval frameworks (e.g. LangGraph, LlamaIndex, CrewAI)
- Use of fine-tuned sentiment or market-focused models
- Show as many llm gen ai tools and frameworks
- Visual dashboards (e.g., Streamlit, Gradio) to inspect LLM decisions or model confidence
- Added domain-specific knowledge graphs or context compression methods

---

Please document any external services used (e.g., OpenAI APIs) and note if you simulated outputs due to token limits or cost.
