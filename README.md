# LLM Engineer Technical Assessment: Financial Trading Intelligence

## Overview
This technical assessment evaluates your ability to build an LLM-powered trading intelligence system using a real-world cryptocurrency dataset. You'll demonstrate expertise in implementing various components of an advanced LLM system for financial trading decision-making.

## Dataset Description
The dataset (`gold-1.csv`) contains the following columns:
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

### Task 2: LLM Trading Flow Design
Develop a chain-of-thought reasoning workflow that:
- Takes current market data as input
- Analyzes sentiment and psychological factors
- Retrieves similar historical patterns
- Generates a structured trading recommendation with confidence score
- Provides explicit reasoning for the decision

### Task 3: Performance Evaluation
Create a backtesting framework that:
- Evaluates the LLM's trading decisions against historical data
- Measures precision, recall, and F1-score of trading signals
- Calculates ROI, Sharpe ratio, and drawdown metrics
- Compares performance against baseline strategies

### Task 4: Hallucination Reduction
Implement techniques to reduce hallucinations in the system by:
- Creating a factual grounding mechanism using historical data
- Implementing confidence scoring for predictions
- Designing guardrails to prevent extreme predictions
- Measuring and minimizing hallucination rates

### Task 5: Experimentation & Optimization
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
- Implementation of agentic structure for autonomous trading decisions
- Integration with real-time market data sources
- Fine-tuning of a specialized model for financial sentiment analysis
- Creative approaches to extract insights from social media signals
