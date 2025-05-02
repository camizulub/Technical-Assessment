# Backtesting Package

A modular Python package for trading strategy development and backtesting with a focus on data validation and visualization. Right now, it is using a vectorized approach.

## Features

- Data validation using Pydantic models
- Trading strategy implementation (SMA/EMA crossover, Bollinger Bands, or Custom Strategies)
- Trade generation and analysis
- Performance metrics calculation
- Visualization with Plotly

## Installation

```bash
pip install -e .
```

## Usage Example

```python
import pandas as pd
from trading_package.models import MarketData
from trading_package.strategies import CrossoverStrategies, TradeGenerator
from trading_package.utils import PerformanceCalculator
from trading_package.visualization import TradingVisualizer

# Load market data
df = pd.read_csv('market_data.csv')
market_data = MarketData(df=df)

# Create a strategy
sma_strategy = CrossoverStrategies.sma_crossover(
    market_data=market_data,
    short_window=6,
    long_window=24,
    strategy_name="SMA_Crossover"
)

# Generate trades
sma_trades = TradeGenerator.generate_trade_dataframe(
    strategy_signals=sma_strategy
)

# Calculate performance metrics
sma_metrics = PerformanceCalculator.calculate_performance_metrics(
    trades_data=sma_trades,
    market_data = sma_strategy
)

# Display full metrics
print(pd.Series(sma_metrics.metrics))

# Display key metrics
print("\nKey metrics for SMA strategy:")
sma_key_metrics = PerformanceCalculator.get_key_metrics(sma_metrics)
for key, value in sma_key_metrics.items():
    print(f"  {key}: {value}")

# Visualize SMA trades
sma_fig = TradingVisualizer.plot_trading_signals(
    strategy_signals=sma_strategy,
    title=f"SMA Crossover Strategy",
)
sma_fig.show()

# Plot equity curve
sma_equity_curve = PerformanceCalculator.get_equity_curves(sma_strategy, sma_trades)
equity_fig = TradingVisualizer.plot_equity_curves(
    equity_curves=[
        sma_equity_curve,
    ],
    title="Equity Curve",
    dark_mode=True,
    width=1000,
    height=600,
)
equity_fig.show()

```

## Package Structure

- `models/`: Pydantic data models
- `strategies/`: Trading strategy implementations
- `utils/`: Statistical and performance calculation utilities
- `visualization/`: Data visualization tools

## Requirements

- Python 3.8+
- pandas
- numpy
- plotly
- pydantic
