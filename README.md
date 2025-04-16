# Trading Signal Processing System - Technical Assessment

## Introduction
This assessment evaluates your ability to build a trading signal processing system that processes pre-generated trading signals and executes trading decisions based on those signals. You are NOT required to develop or create the trading signals themselves - these will be provided to your system. The assessment is structured in two clear phases with focused deliverables.

## Environment
Choose a cloud environment:
- Google Cloud Platform (Free tier available: $300 credit) https://cloud.google.com/free?hl=en 
- AWS (Free tier available)

## Dataset
The provided dataset (`gold.csv`) contains cryptocurrency trading data with pre-generated signals:
- `datetime`: Timestamp of the trading signal
- `close`: Closing price of Bitcoin
- Technical indicators (e.g., `mvrv_btc_momentum`, `spot_volume_daily_sum`)
- Sentiment analysis features (e.g., `sentiment`, `dominant_emotions`)
- `Signal`: Pre-generated trading action to execute (BUY (1), SELL (-1), HOLD (0))

**Important:** Your task is to process these pre-generated signals, not to create new trading signals or develop a trading strategy algorithm.

## Assessment Structure

### Phase 1: Core Signal Processing Service
Build a production-ready service that processes trading signals and calculates performance.

**Requirements:**
1. Create a REST API with the following endpoints:
   - `POST /signal` - Process a new pre-generated trading signal (you don't create the signal value)
   - `GET /performance` - Get current portfolio performance
   - **Bonus Points:** Use Pydantic for data validation

2. Implement signal processing logic:
   - Process pre-generated signals sequentially by timestamp
   - Execute positions based on the provided signals (BUY, SELL, HOLD)
   - Calculate returns after each executed signal
   - Track cumulative performance

3. Production requirements:
   - Containerized application (Docker)
   - Deployed to Google Cloud Run or AWS Lambda
   - Comprehensive error handling
   - Basic logging
   - Unit tests for core functionality
   
4. Cloud storage requirements:
   - Portfolio performance data must be stored in a cloud service:
     - Google Cloud: BigQuery or Cloud Storage
     - AWS: S3 or DynamoDB
   - Historical signal processing results should be queryable

**Deliverables:**
- Source code with clear structure
- Dockerfile
- README with setup instructions and API documentation
- Screenshots of deployed service in Google Cloud Run or AWS Lambda
- Access information for the cloud storage containing portfolio data
- Test results showing baseline performance on the full dataset

### Phase 2: Performance Enhancement (This can be done in a notebook or VM, no need to deploy)
Implement an intelligent enhancement using generative AI to improve trading performance.

**Requirements:**
1. Build a memory mechanism that (Use an llm - Gemini flash API is free):
   - Tracks historical signal accuracy 
   - Identifies patterns in successful/unsuccessful signals
   - Makes intelligent override decisions

2. Compare performance:
   - Run baseline and enhanced systems in parallel
   - Generate clear performance metrics
   - Document improvements

**Deliverables:**
- Enhanced source code
- Documentation of enhancement approach
- Performance comparison analysis

## Submission Guidelines
1. Provide a GitHub repository with your solution
2. Include a README with:
   - Setup instructions
   - API documentation
   - Architecture overview
   - Cloud deployment instructions
   - Cloud storage access details
   - Performance analysis
   - Any assumptions made
3. Provide screenshots of the deployed service in Google Cloud Run or AWS Lambda
4. Share access credentials for the cloud storage (or instructions to generate access)

## Testing Your Solution
We will:
1. Deploy your container using the provided Dockerfile
2. Test the API by sending sequential pre-generated signals from our dataset
3. Verify that your system correctly processes these signals and calculates performance
4. Evaluate the enhancement approach

Note: We're testing your ability to build a system that consumes and processes signals, not your ability to generate trading signals.

## Tips for Success
- Focus on building a robust core system before moving to enhancements
- Prioritize code quality and error handling
- Document your approach clearly
- Keep the solution focused on the requirements
