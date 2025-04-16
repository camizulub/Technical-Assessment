# Trading Signal Processing System - Technical Assessment

## Overview
Build a production-ready system that processes trading signals and implements an intelligent memory-based override mechanism to improve trading performance.

## Dataset Description
The provided dataset (`gold.csv`) contains cryptocurrency trading data with the following key components:

### Core Trading Data
- `datetime`: Timestamp of the trading signal
- `close`: Closing price of Bitcoin
- `label_next_bar`: Target label for the next bar
- `best_label`: Optimal label for the current bar

### Technical Indicators
- `mvrv_btc_momentum`: Bitcoin momentum indicator
- `spot_volume_daily_sum`: Daily trading volume

### Sentiment Analysis Features
- `sentiment`: Overall market sentiment score
- `dominant_emotions`: Primary emotions detected in market sentiment
- `psychology_explanation`: Detailed psychological analysis of market conditions
- `intensity`: Sentiment intensity score

### News and Market Analysis
- `next_news_prediction`: Predicted impact of upcoming news
- `summary`: Market summary and analysis
- `key_factors`: Important market drivers
- `dominant_sentiment`: Overall market sentiment classification

## Requirements

### Part 1: Core Signal Processing Service
- Build a service that processes trading signals sequentially by timestamp
- Implement position taking based on signals
- Calculate returns and track cumulative performance
- Must be production-ready with:
  - Comprehensive error handling
  - Unit and integration tests
  - Docker containerization

### Part 2: Intelligent Signal Enhancement
- Implement memory mechanism to track signal accuracy
  - BUY signals considered correct if return > 0
  - SELL signals considered correct if return < 0
- Create override logic for signals with poor recent performance
- Consider feature similarity when evaluating past performance
- Run baseline and enhanced systems in parallel
- Compare and report performance differences

## Testing Requirements

### 1. Signal Processing Tests
- Sequential processing of signals by timestamp
- Position taking logic validation
- Return calculation accuracy
- Handling of missing/invalid data

### 2. Memory Mechanism Tests
- Signal accuracy tracking
- Override logic validation
- Feature similarity calculations
- Memory-based decision making

### 3. Performance Comparison Tests
- Parallel execution validation
- Performance metrics calculation
- Comparison reporting accuracy

### 4. Error Handling Tests
- Invalid timestamp handling
- Missing price data handling
- Invalid signal type handling
- Edge case handling (first/last signals)

### 5. Data Validation Tests
- Data type validation
- Range validation for numerical fields
- Timestamp format validation
- Required field presence validation

## Deliverables
1. Complete source code with tests
2. Dockerfile for containerization
3. README with:
   - Setup instructions
   - System architecture overview
   - Performance analysis and comparison
   - API documentation
   - Testing strategy

## Technical Requirements

### Core Components
- Signal processing engine
- Position management system
- Performance calculation module
- Memory-based override mechanism
- Parallel execution framework
- Monitoring and logging system

### Data Requirements
- CSV file with sample trading signals containing:
  - Timestamp
  - Signal type (BUY/SELL)
  - Price data
  - Relevant features for similarity analysis

### Documentation Requirements
- System architecture diagrams
- Setup and deployment guides
- Performance analysis reports

## Evaluation Criteria
1. Code quality and organization
2. System design and architecture
3. Testing coverage and methodology
4. Performance improvements achieved
5. Documentation clarity
6. Production readiness

## Bonus Points
- Build an agent that stores the signals and the past performance in order to improve the current signals
- Real-time processing capabilities
- Scalability considerations
- Advanced feature similarity metrics
- Visualization of performance comparisons
- CI/CD pipeline setup

## Testing Framework
The solution should include a comprehensive test suite using either:
- Python's unittest framework
- pytest framework

Tests should be organized into logical categories and include:
- Unit tests for individual components
- Integration tests for system interactions
- Performance tests for parallel processing
- Data validation tests
- Error handling tests

Each test should be well-documented with:
- Clear test objectives
- Expected outcomes
- Edge cases considered
- Error scenarios handled
