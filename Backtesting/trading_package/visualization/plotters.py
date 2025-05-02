"""
Visualization tools for trading data and performance analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List

from ..models.data_models import MarketData, StrategySignals, PerformanceMetrics


class TradingVisualizer:
    """
    Class for creating visualizations of trading data and performance metrics.
    """
    
    @staticmethod
    def plot_trading_signals(
        strategy_signals: StrategySignals,
        title: str = "Trading Strategy Performance",
        log_scale: bool = False, 
        dark_mode: bool = True,
    ) -> go.Figure:
        """
        Create a comprehensive plot showing price data, signals, and trades.
        
        Parameters
        ----------
        strategy_signals : StrategySignals
            Strategy signals with position information
        title : str, default "Trading Strategy Performance"
            Title for the plot
        log_scale : bool, optional
            Whether to use logarithmic scale for y-axis, default True
        dark_mode : bool, optional
            Whether to use dark mode theme, default True
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        signals_df = strategy_signals.df.copy()
        

        # Identify buy and sell signals
        buy_signals = signals_df[(signals_df['position'] > 0) & 
                                (signals_df['position'].shift(1) <= 0)]
        sell_signals = signals_df[(signals_df['position'] < 0) & 
                                (signals_df['position'].shift(1) >= 0)]
        
        # Create figure
        fig = go.Figure()
        
        # Add the price line
        fig.add_trace(go.Scatter(
            x=signals_df['datetime_est_10mins'], 
            y=signals_df['close'], 
            mode='lines', 
            name='Price', 
            showlegend=False,
            line=dict(color='#F2A900', width=2)
        ))
        
        # Add buy signals
        fig.add_trace(go.Scatter(
            x=buy_signals['datetime_est_10mins'],
            y=buy_signals['close'],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-right'),
            name='Buy Signal'
        ))
        
        # Add sell signals
        fig.add_trace(go.Scatter(
            x=sell_signals['datetime_est_10mins'],
            y=sell_signals['close'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-left'),
            name='Sell Signal'
        ))
        
        # Configure layout
        template = 'plotly_dark' if dark_mode else 'plotly_white'
        
        fig.update_layout(
            title=f'<b>{title}<b>',
            xaxis_title='<b>Date<b>', 
            yaxis=dict(
                title=dict(text="<b>Price<b>"),
                type="log" if log_scale else "linear"
            ), 
            template=template,
            hoverlabel=dict(
                bgcolor="white",
                font_size=14, 
                font_family="Arial",
                font_color="black",
                bordercolor="darkgray",
            )
        )
        
        return fig
    
    @staticmethod
    def plot_performance_comparison(
        performance_metrics_list: List[PerformanceMetrics],
        metrics_to_plot: Optional[List[str]] = None,
        title: str = "Strategy Performance Comparison",
        dark_mode: bool = True,
        width: int = 1000,
        height: int = 600
    ) -> go.Figure:
        """
        Create a bar chart comparing key performance metrics across strategies.
        
        Parameters
        ----------
        performance_metrics_list : List[PerformanceMetrics]
            List of performance metrics for different strategies
        metrics_to_plot : Optional[List[str]], default None
            List of metric names to include in the comparison.
            If None, includes: ['Return [%]', 'Sharpe Ratio', 'Sortino Ratio', 'Max. Drawdown [%]']
        title : str, default "Strategy Performance Comparison"
            Title for the plot
        dark_mode : bool, optional
            Whether to use dark mode theme, default True
        width : int, default 1000
            Width of the plot in pixels
        height : int, default 600
            Height of the plot in pixels
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        if not performance_metrics_list:
            # Return empty figure if no metrics provided
            return go.Figure()
        
        # Default metrics to plot if not specified
        if not metrics_to_plot:
            metrics_to_plot = ['Return [%]', 'Sharpe Ratio', 'Sortino Ratio', 'Max. Drawdown [%]']
        
        # Extract strategy names and metrics
        strategy_names = [p.strategy_name for p in performance_metrics_list]
        
        # Create a figure with a subplot for each metric
        fig = make_subplots(
            rows=len(metrics_to_plot),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=metrics_to_plot
        )
        
        # Add a bar chart for each metric
        for i, metric in enumerate(metrics_to_plot, 1):
            metric_values = [p.metrics[metric] if metric in p.metrics.keys() else np.nan for p in performance_metrics_list]
            
            # Determine color based on metric type (higher is better, except for drawdown)
            if metric == 'Max. Drawdown [%]':
                colors = ['red' if v < 0 else 'green' for v in metric_values]
            else:
                colors = ['green' if v > 0 else 'red' for v in metric_values]
            
            fig.add_trace(
                go.Bar(
                    x=strategy_names,
                    y=metric_values,
                    name=metric,
                    marker_color=colors,
                    hovertemplate='%{x}<br>' + metric + ': %{y:.2f}<extra></extra>'
                ),
                row=i, col=1
            )
            
        # Configure layout
        template = 'plotly_dark' if dark_mode else 'plotly_white'
        
        # Update layout
        fig.update_layout(
            title=f'<b>{title}<b>',
            showlegend=False,
            template=template,
            width=width,
            height=height * len(metrics_to_plot) / 4,
            hovermode="x"
        )
        
        return fig
    
    @staticmethod
    def plot_equity_curves(
        equity_curves: List[pd.DataFrame],
        title: str = "Equity Curves Comparison",
        dark_mode: bool = True,
        width: int = 1000,
        height: int = 600,
        normalize: bool = True
    ) -> go.Figure:
        """
        Plot equity curves for multiple strategies.
        
        Parameters
        ----------
        equity_curves : List[pd.DataFrame]
            List of equity curves for different strategies
        title : str, default "Equity Curves Comparison"
            Title for the plot
        dark_mode : bool, optional
            Whether to use dark mode theme, default True
        width : int, default 1000
            Width of the plot in pixels
        height : int, default 600
            Height of the plot in pixels
        normalize : bool, default True
            Whether to normalize curves to start from same point
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        fig = go.Figure()
        
        # Check if inputs are valid
        if not equity_curves:
            return fig
        
        # Find common start date
        common_start_date = max(equity_curves[0].index.min() for equity_curve in equity_curves)
        
        # Add trace to figure
        for equity_curve in equity_curves:
            # Slice data from common start date
            curve_data = equity_curve[equity_curve.index >= common_start_date].copy()
            
            if normalize:
                # Normalize to start at 1.0
                initial_value = curve_data['cstrategy'].iloc[0]
                curve_data['cstrategy'] = curve_data['cstrategy'] / initial_value
                
            fig.add_trace(
                go.Scatter(
                    x=curve_data.index,
                    y=curve_data['cstrategy'],
                    name=curve_data['strategy_name'].iloc[0],
                    mode="lines",
                    hovertemplate='%{x}<br>' + curve_data['strategy_name'].iloc[0] + ': %{y:.2%}<extra></extra>'
                )
            )
        
        # Add benchmark trace
        benchmark_data = equity_curves[0][equity_curves[0].index >= common_start_date].copy()
        if normalize:
            initial_benchmark = benchmark_data['creturns'].iloc[0]
            benchmark_data['creturns'] = benchmark_data['creturns'] / initial_benchmark
            
        fig.add_trace(
            go.Scatter(
                x=benchmark_data.index,
                y=benchmark_data['creturns'],
                name="Benchmark (Buy & Hold)",
                mode="lines",
                line=dict(color='#F2A900', dash='dash'),
                hovertemplate='%{x}<br>Benchmark: %{y:.2%}<extra></extra>'
            )
        )
        
        # Configure layout
        template = 'plotly_dark' if dark_mode else 'plotly_white'
        
        # Update layout
        y_title = "<b>Normalized Performance</b>" if normalize else "<b>Performance ($)</b>"
        fig.update_layout(
            title=f'<b>{title}<b>',
            xaxis_title="<b>Date<b>",
            yaxis_title=y_title,
            template=template,
            hovermode="x unified",
            width=width,
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
