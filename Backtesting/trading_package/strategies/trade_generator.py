"""
Trade generation and analysis functionality.
"""

import pandas as pd

from ..models.data_models import StrategySignals, TradeData


class TradeGenerator:
    """
    Class for generating and analyzing trades from strategy signals.
    """
    
    @staticmethod
    def generate_trade_dataframe(strategy_signals: StrategySignals, price_col: str = 'close') -> TradeData:
        """
        Convert position data to a detailed trade-by-trade history dataframe.
        
        Analyzes a time series of positions and prices to identify individual trades,
        calculating entry/exit prices, profit/loss, and trade durations.
        
        Parameters
        ----------
        strategy_signals : StrategySignals
            Strategy signals with position information
        price_col : str, default 'close'
            Column name containing the price data
            
        Returns
        -------
        TradeData
            DataFrame with one row per completed trade
        """
        # Input validation
        required_columns = ['position', price_col, 'datetime_est_10mins']
        df = strategy_signals.df
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Make a copy to avoid modifying original DataFrame
        df = df.copy()
        
        # Identify where positions change (entry/exit points)
        position_changes = df['position'].diff().fillna(df['position'])
        
        # Initialize containers for trade data
        trades = []
        
        # State variables for tracking trades
        current_position = 0
        entry_bar = None
        entry_price = None
        entry_time = None
        
        # Process each bar to identify trades
        for idx, row in df.iterrows():
            # Get timestamp or use index if datetime column doesn't exist
            current_time = row['datetime_est_10mins']
            
            # Check if position has changed at this bar
            if position_changes.loc[idx] != 0:
                # Case 1: New position (entry from flat)
                if current_position == 0 and row['position'] != 0:
                    entry_bar = idx
                    entry_price = row[price_col]
                    entry_time = current_time
                    current_position = row['position']
                    
                # Case 2: Position exit or reversal
                elif current_position != 0 and (
                        row['position'] == 0 or 
                        (current_position > 0 and row['position'] < 0) or 
                        (current_position < 0 and row['position'] > 0)):
                    
                    # Calculate trade results
                    exit_price = row[price_col]
                    pnl = (exit_price - entry_price) * current_position
                    return_pct = pnl / entry_price * abs(current_position)
                    duration = current_time - entry_time
                    
                    # Record completed trade
                    trades.append({
                        'Size': current_position,
                        'EntryBar': entry_bar,
                        'ExitBar': idx,
                        'EntryPrice': entry_price,
                        'ExitPrice': exit_price,
                        'PnL': pnl,
                        'ReturnPct': return_pct,
                        'EntryTime': entry_time,
                        'ExitTime': current_time,
                        'Duration': duration
                    })
                    
                    # Reset position tracking or setup for new position if reversing
                    if row['position'] == 0:
                        current_position = 0
                    else:
                        # For position reversal, immediately setup new trade
                        entry_bar = idx
                        entry_price = row[price_col]
                        entry_time = current_time
                        current_position = row['position']
        
        # Handle open position at end of data
        if current_position != 0:
            last_row = df.iloc[-1]
            last_time = last_row['datetime_est_10mins']
            
            # Calculate results for open position
            exit_price = last_row[price_col]
            pnl = (exit_price - entry_price) * abs(current_position)
            return_pct = pnl / entry_price
            duration = last_time - entry_time
            
            # Add the open position as a "closed" trade
            trades.append({
                'Size': current_position,
                'EntryBar': entry_bar,
                'ExitBar': df.index[-1],
                'EntryPrice': entry_price,
                'ExitPrice': exit_price,
                'PnL': pnl,
                'ReturnPct': return_pct,
                'EntryTime': entry_time,
                'ExitTime': last_time,
                'Duration': duration
            })
        
        # Create DataFrame from trades list
        if not trades:
            # Return empty DataFrame with appropriate columns if no trades found
            return TradeData(df=pd.DataFrame(columns=[
                'Size', 'EntryBar', 'ExitBar', 'EntryPrice', 'ExitPrice',
                'PnL', 'ReturnPct', 'EntryTime', 'ExitTime', 'Duration'
            ]))
        
        return TradeData(df=pd.DataFrame(trades))
