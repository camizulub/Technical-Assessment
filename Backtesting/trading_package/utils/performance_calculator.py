"""
Performance calculation and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
pd.set_option('future.no_silent_downcasting', True)

from ..models.data_models import TradeData, PerformanceMetrics, MarketData
from ..utils.statistics import StatisticsUtils
from typing import Optional, List

class PerformanceCalculator:
    """
    Class for calculating and analyzing trading performance metrics.
    """
    
    @staticmethod
    def get_equity_curves(market_data: MarketData, trades_data: TradeData, initial_capital: float = 10000) -> pd.DataFrame:
        """
        Extract the equity curves from the market and trades data.
        
        Parameters
        ----------
        market_data : MarketData
            Market data
        trades_data : TradeData
            Trades data
        initial_capital : float, default 10000
            Initial capital for equity curve calculation
            
        Returns
        -------
        pd.DataFrame
            DataFrame with equity curves
        """
        # Extract data
        trades_df = trades_data.df
        price_df = market_data.df
        
        # === Equity curve calculation ===
        price_df = price_df.set_index('datetime_est_10mins').loc[
            trades_df['EntryTime'].min():]
        equity = pd.DataFrame(index=price_df.index)
        
        equity['strategy_name'] = price_df['strategy_name'].iloc[0] if 'strategy_name' in price_df.columns else 'Unknown Strategy'
        # Calculate daily log returns and strategy returns
        equity['returns'] = np.log(price_df['close'] / price_df['close'].shift(1))
        equity['strategy'] = price_df['position'].shift(1) * equity['returns']
        
        # Convert log returns to cumulative returns
        equity['creturns'] = equity['returns'].cumsum().apply(np.exp)
        equity['cstrategy'] = equity['strategy'].cumsum().apply(np.exp)
        
        # Convert to cash values
        equity['strategy_cash'] = equity['cstrategy'] * initial_capital
        equity['returns_cash'] = equity['creturns'] * initial_capital
        equity.dropna(inplace=True)
        
        return equity
    
    @staticmethod
    def calculate_performance_metrics(
        trades_data: TradeData,
        market_data: MarketData,
        risk_free_rate: float = 0.0,
        initial_capital: float = 10000
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive trading strategy performance metrics.
        
        Parameters
        ----------
        trades_data : TradeData
            Trade information
        market_data : MarketData
            Market price data
        risk_free_rate : float, default 0.0
            Annual risk-free rate in decimal (e.g., 0.02 for 2%)
        initial_capital : float, default 10000
            Initial capital for equity curve calculation
            
        Returns
        -------
        PerformanceMetrics
            Performance metrics
        """
        # Get strategy name
        strategy_name = market_data.df['strategy_name'].iloc[0] if 'strategy_name' in market_data.df.columns else 'Unknown Strategy'
        
        # Extract data
        trades_df = trades_data.df
        price_df = market_data.df
        
        # Initialize results container
        metrics = {}
        
        # Check if trades dataframe is empty
        if trades_df.empty:
            raise ValueError("No trades data provided")
        
        # Extract trade data
        pl = trades_df['PnL']
        returns = trades_df['ReturnPct']
        
        # === Time period analysis ===
        if 'EntryTime' in trades_df.columns and 'ExitTime' in trades_df.columns:
            trade_times = pd.to_datetime(
                trades_df[['EntryTime', 'ExitTime']].values.flatten())
            metrics['Start'] = trade_times.min()
            metrics['End'] = trade_times.max()
            metrics['Duration'] = metrics['End'] - metrics['Start']
        
        # === Equity curve calculation ===
        equity = PerformanceCalculator.get_equity_curves(market_data, trades_data, initial_capital)
        
        # Calculate exposure time (% of time with open positions)
        exposure_time = (price_df['position'] != 0).mean() * 100
        metrics['Exposure Time [%]'] = exposure_time
        
        # === Basic performance metrics ===
        metrics['Equity Final [$]'] = equity['strategy_cash'].iloc[-1]
        metrics['Equity Peak [$]'] = equity['strategy_cash'].max()
        # metrics['Return [%]'] = (
        #     (equity['cstrategy'].iloc[-1] - equity['cstrategy'].iloc[0]) * 100
        # )
        metrics['Return [%]'] = (
            (equity['cstrategy'].iloc[-1] / equity['cstrategy'].iloc[0] - 1) * 100
        )
        metrics['Buy & Hold Return [%]'] = (
            (equity['creturns'].iloc[-1] / equity['creturns'].iloc[0] - 1) * 100
        )
        
        # === Determine trading frequency ===
        # Calculate median difference between consecutive index points
        time_diff = pd.Series(equity.index).diff().median()
        freq_days = time_diff.days
        freq_seconds = time_diff.total_seconds()
        
        # Check if dataset includes weekends
        have_weekends = (
            equity.index.dayofweek.to_series().between(5, 6).mean() > 2/7 * 0.6
        )
        
        # Set appropriate number of trading days/periods per year based on data frequency
        if freq_days >= 7:
            if freq_days == 7:
                annual_trading_days = 52  # Weekly data
            elif freq_days == 31:
                annual_trading_days = 12  # Monthly data 
            elif freq_days == 365:
                annual_trading_days = 1   # Annual data
            else:
                annual_trading_days = 365 if have_weekends else 252  # Daily data
            freq = {7: 'W', 31: 'ME', 365: 'YE'}.get(freq_days, 'D')
        else:
            # Handle intraday frequencies
            if freq_seconds <= 60:  # 1 minute or less
                annual_trading_days = 252 * 6.5 * 60  # Approx trading minutes per year
                freq = 'T'
            elif freq_seconds <= 3600:  # Hourly
                annual_trading_days = 252 * 6.5  # Trading hours per year
                freq = 'H'
            else:  # Default to daily
                annual_trading_days = 252
                freq = 'D'
        
        # === Calculate return statistics ===
        # Resample to appropriate frequency and calculate returns
        day_returns = equity['strategy_cash'].resample(freq.lower()).last().dropna().pct_change()
        gmean_day_return = StatisticsUtils.geometric_mean(day_returns)
        
        # Calculate annualized metrics
        annualized_return = (1 + gmean_day_return)**annual_trading_days - 1
        metrics['Return (Ann.) [%]'] = annualized_return * 100
        
        # Calculate annualized volatility
        vol_term1 = day_returns.var(ddof=int(bool(day_returns.shape)))
        vol_term2 = (1 + gmean_day_return)**2
        vol_annualized = np.sqrt(
            (vol_term1 + vol_term2)**annual_trading_days - 
            (1 + gmean_day_return)**(2 * annual_trading_days)
        )
        metrics['Volatility (Ann.) [%]'] = vol_annualized * 100
        
        # Calculate CAGR if duration > 0
        # For intraday data, calculate years including trading hours
        trading_hours_per_day = 6.5  # Standard market hours
        time_in_years = (
            (metrics['Duration'].days + 
             metrics['Duration'].seconds / (86400 * trading_hours_per_day/24)) / 
            252  # Use trading days instead of calendar days
        )
        if time_in_years > 0:
            cagr = (
                (metrics['Equity Final [$]'] / 
                 equity['strategy_cash'].iloc[0])**(1 / time_in_years) - 1
            )
            metrics['CAGR [%]'] = cagr * 100
        else:
            metrics['CAGR [%]'] = np.nan
        
        # === Risk-adjusted return metrics ===
        # Sharpe ratio calculation
        sharpe_denominator = metrics['Volatility (Ann.) [%]'] or np.nan
        metrics['Sharpe Ratio'] = (
            (metrics['Return (Ann.) [%]'] - risk_free_rate * 100) / 
            sharpe_denominator
        )
        
        # Sortino ratio calculation (using downside deviation)
        with np.errstate(divide='ignore'):
            downside_returns = day_returns.clip(-np.inf, 0)
            downside_deviation = np.sqrt(np.mean(downside_returns**2))
            downside_deviation_ann = downside_deviation * np.sqrt(annual_trading_days)
            
            sortino_denominator = downside_deviation_ann or np.nan
            metrics['Sortino Ratio'] = (
                (annualized_return - risk_free_rate) / sortino_denominator
            )
        
        # === Drawdown analysis ===
        # Calculate drawdown series
        dd = 1 - equity['strategy_cash'] / np.maximum.accumulate(
            equity['strategy_cash'])
        
        # Find drawdown periods
        is_drawdown = dd > 0
        dd_start = (is_drawdown) & (~is_drawdown.shift(1).fillna(False))
        dd_end = (~is_drawdown) & (is_drawdown.shift(1).fillna(False))
        
        # Handle ongoing drawdown at the end of the series
        if is_drawdown.iloc[-1]:
            dd_end.iloc[-1] = True
        
        # Calculate drawdown durations
        start_idx = equity.index[dd_start]
        end_idx = equity.index[dd_end]
        
        dd_durations = []
        for i in range(min(len(start_idx), len(end_idx))):
            duration = end_idx[i] - start_idx[i]
            dd_durations.append(duration)
        
        dd_dur = pd.Series(dd_durations)
        
        # Calculate drawdown statistics
        max_dd = dd.max()
        max_dd_duration = dd_dur.max() if len(dd_dur) > 0 else pd.Timedelta(0)
        avg_dd_duration = dd_dur.mean() if len(dd_dur) > 0 else pd.Timedelta(0)
        
        # Calculate average drawdown magnitude
        non_zero_dd = dd[is_drawdown]
        avg_drawdown = non_zero_dd.mean() if len(non_zero_dd) > 0 else 0
        
        # Calmar ratio (annualized return / maximum drawdown)
        metrics['Calmar Ratio'] = annualized_return / (max_dd or np.nan)
        
        # === Alpha and Beta calculation ===
        # Extract log returns for strategy and market
        equity_log_returns = equity['strategy']
        market_log_returns = equity['returns']
        
        # Filter out NaN values for accurate covariance calculation
        valid_mask = ~np.isnan(equity_log_returns) & ~np.isnan(market_log_returns)
        eq_returns_clean = equity_log_returns[valid_mask]
        mkt_returns_clean = market_log_returns[valid_mask]
        
        # Calculate beta using covariance/variance
        beta = np.nan
        if len(eq_returns_clean) > 1 and len(mkt_returns_clean) > 1:
            cov_matrix = np.cov(eq_returns_clean, mkt_returns_clean)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        # Calculate Jensen's Alpha (risk-adjusted excess return)
        ann_market_return = (
            metrics['Buy & Hold Return [%]'] / time_in_years
        ) if time_in_years else 0
        
        metrics['Alpha [%]'] = (
            metrics['Return (Ann.) [%]'] - 
            risk_free_rate * 100 - 
            beta * (ann_market_return - risk_free_rate * 100)
        )
        metrics['Beta'] = beta
        
        # Store drawdown metrics
        metrics['Max. Drawdown [%]'] = -max_dd * 100
        metrics['Avg. Drawdown [%]'] = -avg_drawdown * 100
        metrics['Max. Drawdown Duration'] = max_dd_duration
        metrics['Avg. Drawdown Duration'] = avg_dd_duration
        
        # === Trade-specific metrics ===
        n_trades = len(trades_df)
        metrics['# Trades'] = n_trades
        
        if n_trades > 0:
            # Calculate win rate
            win_rate = (pl > 0).mean()
            metrics['Win Rate [%]'] = win_rate * 100
            
            # Trade performance extremes
            metrics['Best Trade [%]'] = returns.max() * 100
            metrics['Worst Trade [%]'] = returns.min() * 100
            
            # Calculate average trade return using geometric mean
            if all(returns >= -1):  # Ensure no returns are < -100%
                mean_return = np.exp(np.mean(np.log(1 + returns))) - 1
                metrics['Avg. Trade [%]'] = mean_return * 100
            
            # Trade duration statistics if available
            if 'Duration' in trades_df.columns:
                metrics['Max. Trade Duration'] = trades_df['Duration'].max()
                metrics['Avg. Trade Duration'] = trades_df['Duration'].mean()
            
            # Profit factor (sum of profits / sum of losses)
            win_sum = returns[returns > 0].sum()
            loss_sum = abs(returns[returns < 0].sum())
            metrics['Profit Factor'] = win_sum / (loss_sum or np.nan)
            
            # Expectancy (average trade return)
            metrics['Expectancy [%]'] = returns.mean() * 100
            
            # System Quality Number (mean trade profit / std deviation * sqrt(n_trades))
            metrics['SQN'] = (
                np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)
            )
            
            # Kelly Criterion (optimal position sizing)
            if win_rate > 0 and len(pl[pl < 0]) > 0 and pl[pl < 0].mean() != 0:
                avg_win = pl[pl > 0].mean()
                avg_loss = abs(pl[pl < 0].mean())
                metrics['Kelly Criterion'] = (
                    win_rate - (1 - win_rate) / (avg_win / avg_loss)
                )
        
        return PerformanceMetrics(metrics=metrics, strategy_name=strategy_name)
    
    @staticmethod
    def get_key_metrics(
        performance_metrics: PerformanceMetrics,
        metrics_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract specified metrics from the full performance metrics.
        
        Parameters
        ----------
        performance_metrics : PerformanceMetrics
            Full performance metrics
        metrics_list : Optional[List[str]], default None
            List of metric names to extract. If None, returns default key metrics.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with requested metrics
        """
        metrics = performance_metrics.metrics
        
        # Default key metrics if no list provided
        default_metrics = [
            'Strategy',
            'Return [%]',
            'CAGR [%]',
            'Sharpe Ratio', 
            'Sortino Ratio',
            'Max. Drawdown [%]',
            'Win Rate [%]',
            '# Trades',
            'Profit Factor'
        ]
        
        # Use provided metrics list or default
        metrics_to_extract = metrics_list if metrics_list is not None else default_metrics
        
        # Extract requested metrics
        key_metrics = {}
        for metric in metrics_to_extract:
            if metric == 'Strategy':
                key_metrics[metric] = performance_metrics.strategy_name
            else:
                key_metrics[metric] = metrics.get(metric, np.nan)
        
        return key_metrics
