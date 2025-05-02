import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class Backtester:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0):
        """
        Initialize the backtester with market data and capital.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing market data with at least 'datetime' and 'close' columns
        initial_capital : float, default 10000.0
            Starting capital for the backtest
        """
        # Validate input data
        required_cols = ['datetime', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.results = None
        
    def run(self, positions: pd.Series, position_col: str = 'position', 
            commission: float = 0.001, slippage: float = 0.0) -> pd.DataFrame:
        """
        Run backtest with given positions.
        
        Parameters
        ----------
        positions : pd.Series
            Series with position signals (1=long, -1=short, 0=neutral)
        position_col : str, default 'position'
            Name of the column with position data
        commission : float, default 0.001
            Commission rate as a percentage (0.001 = 0.1%)
        slippage : float, default 0.0
            Slippage as a percentage (0.001 = 0.1%)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with backtest results
        """
        # Create a results DataFrame
        results = self.data.copy()
        
        # Add position column
        results[position_col] = positions
        
        # Calculate returns based on close prices
        results['market_return'] = results['close'].pct_change()
        
        # Calculate returns from positions (previous position * current market return)
        results['strategy_return'] = results[position_col].shift(1) * results['market_return']
        
        # Apply commission and slippage when position changes
        position_changes = results[position_col].diff().abs()
        results['costs'] = position_changes * (commission + slippage)
        results['strategy_return'] = results['strategy_return'] - results['costs']
        
        # Calculate cumulative returns
        results['market_cumulative_return'] = (1 + results['market_return']).cumprod() - 1
        results['strategy_cumulative_return'] = (1 + results['strategy_return']).cumprod() - 1
        
        # Calculate equity curve
        results['market_equity'] = self.initial_capital * (1 + results['market_cumulative_return'])
        results['strategy_equity'] = self.initial_capital * (1 + results['strategy_cumulative_return'])
        
        # Store results
        self.results = results
        
        return results
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with performance metrics
        """
        if self.results is None:
            raise ValueError("Run backtest before calculating metrics")
        
        # Extract relevant data
        returns = self.results['strategy_return'].dropna()
        
        # Calculate metrics
        total_return = self.results['strategy_cumulative_return'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Calculate Sharpe ratio (annualized)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = self.results['strategy_cumulative_return']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        wins = (returns > 0).sum()
        losses = (returns < 0).sum()
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Calculate profit factor
        total_profit = returns[returns > 0].sum()
        total_loss = abs(returns[returns < 0].sum())
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
        
        # Calculate signal metrics if actual signals exist
        signal_metrics = {}
        if 'Signal' in self.results.columns:
            # Convert positions to signals (1 = buy, -1 = sell)
            pred_signals = np.sign(self.results['position'].fillna(0))
            true_signals = self.results['Signal'].fillna(0)
            
            # Filter out zero values (no signal)
            mask = (true_signals != 0)
            
            if mask.sum() > 0:
                # Calculate precision, recall, and F1 score
                y_true = (true_signals[mask] > 0).astype(int)  # Convert to binary (1 = buy, 0 = sell)
                y_pred = (pred_signals[mask] > 0).astype(int)
                
                signal_metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
                signal_metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
                signal_metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Compile all metrics
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            **signal_metrics
        }
        
        return metrics
    
    def plot_results(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot backtest results.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default (12, 8)
            Figure size (width, height)
        """
        if self.results is None:
            raise ValueError("Run backtest before plotting results")
        
        plt.figure(figsize=figsize)
        
        # Set style
        sns.set_style('darkgrid')
        
        # Plot equity curves
        plt.subplot(2, 1, 1)
        plt.plot(self.results['datetime'], self.results['market_equity'], label='Market', alpha=0.7)
        plt.plot(self.results['datetime'], self.results['strategy_equity'], label='Strategy', linewidth=1.5)
        plt.title('Equity Curve')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        cumulative_returns = self.results['strategy_cumulative_return']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        
        plt.fill_between(self.results['datetime'], drawdown, 0, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.ylabel('Drawdown')
        plt.xlabel('Date')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix for signal prediction if actual signals exist.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default (8, 6)
            Figure size (width, height)
        """
        if self.results is None or 'Signal' not in self.results.columns:
            print("No actual signals found in data")
            return
        
        # Convert positions to signals (1 = buy, -1 = sell)
        pred_signals = np.sign(self.results['position'].fillna(0))
        true_signals = self.results['Signal'].fillna(0)
        
        # Filter out zero values (no signal)
        mask = (true_signals != 0)
        
        if mask.sum() > 0:
            # Calculate confusion matrix
            y_true = (true_signals[mask] > 0).astype(int)  # Convert to binary (1 = buy, 0 = sell)
            y_pred = (pred_signals[mask] > 0).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=figsize)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Sell', 'Buy'], 
                        yticklabels=['Sell', 'Buy'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()
        else:
            print("No non-zero signals found in data")
    
    def compare_strategies(self, strategies: Dict[str, pd.Series], 
                           figsize: Tuple[int, int] = (12, 8)) -> pd.DataFrame:
        """
        Compare multiple trading strategies.
        
        Parameters
        ----------
        strategies : Dict[str, pd.Series]
            Dictionary with strategy names as keys and position series as values
        figsize : Tuple[int, int], default (12, 8)
            Figure size (width, height)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with comparison results
        """
        results_list = []
        equity_curves = {}
        
        # Run backtest for each strategy
        for name, positions in strategies.items():
            # Run backtest
            results = self.run(positions)
            
            # Calculate metrics
            metrics = self.calculate_metrics()
            metrics['strategy'] = name
            
            # Store results
            results_list.append(metrics)
            equity_curves[name] = results['strategy_equity']
            
        # Combine results
        comparison = pd.DataFrame(results_list)
        
        # Plot equity curves
        plt.figure(figsize=figsize)
        for name, equity in equity_curves.items():
            plt.plot(self.results['datetime'], equity, label=name)
        
        plt.title('Strategy Comparison')
        plt.ylabel('Portfolio Value')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return comparison.set_index('strategy')

def test_backtester():
    """
    Example usage of the Backtester class.
    """
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    closes = np.random.normal(0, 1, 1000).cumsum() + 100
    
    data = pd.DataFrame({
        'datetime': dates,
        'close': closes
    })
    
    # Create random signals
    signals = pd.Series(np.random.choice([-1, 0, 1], size=1000), index=data.index)
    
    # Initialize backtester
    bt = Backtester(data)
    
    # Run backtest
    results = bt.run(signals)
    
    # Calculate metrics
    metrics = bt.calculate_metrics()
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Plot results
    bt.plot_results()
    
    # Compare strategies
    strategies = {
        'Random': signals,
        'Buy and Hold': pd.Series(1, index=data.index),
        'Sell and Hold': pd.Series(-1, index=data.index)
    }
    
    comparison = bt.compare_strategies(strategies)
    print("\nStrategy Comparison:")
    print(comparison)

if __name__ == "__main__":
    test_backtester() 