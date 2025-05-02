# LLM Engineer Technical Assessment: Financial Trading Intelligence

## Overview
This technical assessment evaluates your ability to build an LLM-powered trading intelligence system using a real-world cryptocurrency dataset. You'll demonstrate expertise in implementing various components of an advanced LLM system for financial trading decision-making.

## Dataset Description
The dataset (`gold.csv`) contains the following columns:
- `datetime`: Timestamp of the trading data
- `close`: Bitcoin closing price
- `mvrv_btc_momentum`: Market Value to Realized Value momentum
- `spot_volume_daily_sum`: Daily trading volume
- `Signal`: Binary trading signal (1 or -1)
- `summary`: News summary
- `next_news_prediction`: Predicted future news
- `sentiment`: Sentiment label (Fear, Greed, etc.)
- `index`: Sentiment score (numeric)
- `key_factors`: Key market factors affecting prices
- `dominant_emotions`: Major emotions in the market
- `dominant_sentiment`: Primary sentiment classification
- `intensity`: Sentiment intensity (1-10)
- `psychology_explanation`: Analysis of market psychology

## Tasks

### Task 1: RAG System Implementation
Design and implement a Retrieval-Augmented Generation system that:
- Creates embeddings for historical market data and news
- Builds a knowledge base of market patterns and trading scenarios
- Implements a retrieval mechanism to find relevant historical parallels
- Integrates retrieved context into the LLM's decision-making process

### Task 2: Hallucination Reduction
Implement techniques to reduce hallucinations in the system by:
- Creating a factual grounding mechanism using historical data
- Implementing confidence scoring for predictions
- Designing guardrails to prevent extreme predictions
- Measuring and minimizing hallucination rates

### Task 3: Experimentation & Optimization
Design an experimentation framework that:
- Tests different prompt structures
- Evaluates various RAG retrieval mechanisms
- Compares different LLM models for trading decisions
- Implements a systematic approach to prompt engineering

## Deliverables
- Python implementation of the RAG system
- LLM trading workflow implementation with chain-of-thought reasoning
- Backtesting framework and evaluation metrics
- Documentation of hallucination reduction techniques
- Experimentation results and optimization findings

## Evaluation Criteria
- Code quality and organization
- Effectiveness of the RAG implementation
- Quality of trading decisions and reasoning
- Robustness of hallucination reduction techniques
- Thoroughness of experimentation and optimization

## Bonus Points
- Fine-tuning of a specialized model for financial sentiment analysis
