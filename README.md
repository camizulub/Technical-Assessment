LLM Engineer Technical Assessment: Financial Trading Intelligence
📍 Overview
This assessment evaluates your ability to build a hybrid AI system that leverages LLMs for financial decision-making in algorithmic trading. You will use a structured crypto dataset (gold.csv) to implement trading logic, evaluate model behavior, and improve system performance based on rigorous analysis.

The goal is to demonstrate your practical skills in LLM orchestration, time series integration, prompt engineering, and model evaluation, with a strong focus on clarity, reliability, and innovation.

📊 Dataset Description
The gold.csv dataset includes structured trading signals and unstructured market sentiment data. Columns:

Column	Description
datetime	Timestamp
close	BTC closing price
mvrv_btc_momentum	MVRV momentum indicator
spot_volume_daily_sum	Daily trading volume
Signal	Ground truth signal (-1 or 1)
summary	News summary at that time
next_news_prediction	Predicted news narrative
sentiment	Market sentiment label (Fear, Greed, etc.)
index	Sentiment index score (numeric)
key_factors	Drivers behind market movements
dominant_emotions	Emotional context of the market
dominant_sentiment	Main sentiment class
intensity	Sentiment intensity score (1–10)
psychology_explanation	Narrative explaining market psychology

✅ Tasks
📌 Task 1: Baseline Trading System
Objective:
Develop an LLM-driven trading algorithm that generates a decision signal ∈ (-1, 1) for each row in the dataset.

Requirements:

The output must be based on the available data without look-ahead bias.

Use LLMs (e.g. GPT-based, Claude, Mistral) for generating trading logic from text fields, optionally combined with structured features (hybrid architecture encouraged).

Integrate embedding models and retrieval-based reasoning if applicable.

Explain your architecture clearly.

Deliverables (in /task_1_baseline/):

A Jupyter Notebook (baseline_strategy.ipynb) that:

Loads the data

Defines your model pipeline

Produces decision outputs

Includes commentary on your design decisions

📌 Task 2: Evaluation Framework
Objective:
Evaluate your trading model's performance using the provided Backtester class and define a framework to systematically assess model decisions and behaviors.

Requirements:

Use the Backtester to simulate PnL and draw performance metrics.

Build an evaluation report that includes:

Win rate, Sharpe ratio, drawdown, and signal quality

Qualitative and quantitative model reasoning analysis

Identify promising improvement directions (e.g., prompt tuning, memory context, model grounding).

Deliverables (in /task_2_evaluation/):

A Jupyter Notebook (evaluation_report.ipynb) that:

Runs the backtester

Includes visualizations of performance (e.g., equity curve)

Breaks down LLM-generated reasoning errors or hallucinations

Proposes at least two improvement directions for Task 3

📌 Task 3: Experimentation & Optimization
Objective:
Based on your findings from Task 2, implement two targeted improvements and measure their impact.

Examples of valid improvements:

Refined prompt engineering for better financial reasoning

Retrieval-Augmented Generation using similar past cases

Confidence scoring or hallucination filtering

Using different LLMs or fine-tuned models

Better input feature engineering

Deliverables (in /task_3_experimentation/):

A Jupyter Notebook (experiments_and_results.ipynb) showing:

Your two chosen improvements

Comparative performance metrics

Commentary on what worked and what didn’t

Final recommendations

📦 Submission Instructions
Submit a GitHub repository structured as follows:

pgsql
Copiar
Editar
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
└── README.md  ← Explain how to run your notebooks and dependencies
Note: Use mock LLM outputs or OpenAI API calls as appropriate (mention cost constraints if applicable). You may simulate inference when needed for reproducibility.

🧪 Evaluation Criteria
Area	What We’re Looking For
Code Quality	Modular, documented, reproducible code
LLM System Design	Clear and creative use of LLMs and hybrid models
Backtest Results	Evidence of reasonable strategy performance and generalization
Evaluation Insight	Depth of insight into model decisions, hallucinations, and performance tradeoffs
Experimentation Rigor	Logical follow-through on improvements, with thoughtful results interpretation
Communication & Clarity	Clean, well-documented notebooks showing your reasoning and results

🌟 Bonus Points
Fine-tuning or retrieval pipelines using tools like LangGraph, LlamaIndex, or CrewAI

Use of vector databases, streaming retrieval, or context compression

A lightweight dashboard (e.g., Streamlit or Gradio) to visualize model decisions

Financial sentiment classifier training or scoring extension
