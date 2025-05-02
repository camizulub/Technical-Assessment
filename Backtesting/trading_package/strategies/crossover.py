"""
Crossover trading strategies.
"""

import pandas as pd

from ..models.data_models import MarketData, StrategySignals


class CrossoverStrategies:
    """
    Class for implementing various crossover trading strategies.
    """
    @staticmethod
    def custom_signals(
        market_data: MarketData,
        price_col: str = 'close',
        signal_col: str = 'signal',
        position_col: str = 'position',
        strategy_name: str = "Custom_Signals"
    ) -> StrategySignals:
        """
        Generate trading signals based on custom signals.
        
        Parameters
        ----------
        market_data : MarketData
            Market data with price information
        price_col : str, default 'close'
            Column name containing the price data
        signal_col : str, default 'signal'
            Column name containing the signal data
        position_col : str, default 'position'
            Column name containing the position data
        strategy_name : str, default "Custom_Signals"
            Name of the strategy
            
        Returns
        -------
        StrategySignals
            DataFrame with strategy signals
        """
        # Input validation
        required_cols = {price_col, signal_col, position_col}
        if not required_cols.issubset(market_data.df.columns):
            raise ValueError(f"Required columns {required_cols} not found in DataFrame")
        
        # Access the dataframe from the MarketData object
        df = market_data.df.copy()

        # Initialize signals dataframe with required columns
        signals_df = pd.DataFrame(index=df.index)
        signals_df['datetime_est_10mins'] = df['datetime_est_10mins']
        signals_df['close'] = df[price_col]
        signals_df['signal'] = df[signal_col]
        signals_df['position'] = df[position_col]
        signals_df['strategy_name'] = strategy_name
        
        # Remove rows with NaN values
        signals_df = signals_df.dropna()
        
        return StrategySignals(df=signals_df)
    
    @staticmethod
    def sma_crossover(
        market_data: MarketData, 
        short_window: int = 8,
        long_window: int = 20, 
        price_col: str = 'close',
        position_size: float = 1.0,
        strategy_name: str = "SMA_Crossover"
    ) -> StrategySignals:
        """
        Generate trading signals based on SMA (Simple Moving Average) crossovers.
        
        This strategy generates:
        - Buy signals when the short-term SMA crosses above the long-term SMA
        - Sell signals when the short-term SMA crosses below the long-term SMA
        
        Parameters
        ----------
        market_data : MarketData
            Market data with price information
        short_window : int, default 8
            Window length for the short-term/faster SMA
        long_window : int, default 20
            Window length for the long-term/slower SMA
        price_col : str, default 'close'
            Column name containing the price data
        position_size : float, default 1.0
            Size of position to take on each signal (1 = 100%)
        strategy_name : str, default "SMA_Crossover"
            Name of the strategy
            
        Returns
        -------
        StrategySignals
            DataFrame with strategy signals
        """
        # Input validation
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        
        if price_col not in market_data.df.columns:
            raise ValueError(f"price_col '{price_col}' not found in DataFrame")
            
        if not 0 < position_size <= 1:
            raise ValueError("position_size must be between 0 and 1")
        
        # Access the dataframe from the MarketData object
        df = market_data.df.copy()
        
        # Calculate the two SMAs
        sma_short_col = f'SMA{short_window}'
        sma_long_col = f'SMA{long_window}'
        
        df[sma_short_col] = df[price_col].rolling(window=short_window).mean()
        df[sma_long_col] = df[price_col].rolling(window=long_window).mean()
        
        # Initialize signals dataframe
        signals_df = pd.DataFrame(index=df.index)
        signals_df['datetime_est_10mins'] = df['datetime_est_10mins']
        signals_df['close'] = df[price_col]
        signals_df['signal'] = 0.0
        
        # Generate buy signals (short SMA crosses above long SMA)
        bull_crossover = (
            (df[sma_short_col] > df[sma_long_col]) & 
            (df[sma_short_col].shift(1) <= df[sma_long_col].shift(1))
        )
        signals_df.loc[bull_crossover, 'signal'] = 1.0
        
        # Generate sell signals (short SMA crosses below long SMA)
        bear_crossover = (
            (df[sma_short_col] < df[sma_long_col]) & 
            (df[sma_short_col].shift(1) >= df[sma_long_col].shift(1))
        )
        signals_df.loc[bear_crossover, 'signal'] = -1.0
        
        # Create position column - maintain position until next signal
        positions = []
        current_position = 0
        
        for signal in signals_df['signal'].values:
            if signal != 0:
                # Scale the position by position_size while maintaining signal direction
                current_position = signal * position_size
            positions.append(current_position)
        
        signals_df['position'] = positions
        signals_df['position'] = signals_df['position'].shift(1)
        # Remove rows with NaN values from SMA calculation
        signals_df = signals_df.dropna()
        # Add strategy name
        signals_df['strategy_name'] = f"{strategy_name}_{short_window}_{long_window}"
        
        return StrategySignals(df=signals_df)
    
    @staticmethod
    def ema_crossover(
        market_data: MarketData, 
        short_window: int = 8,
        long_window: int = 20, 
        price_col: str = 'close',
        position_size: float = 1.0,
        strategy_name: str = "EMA_Crossover"
    ) -> StrategySignals:
        """
        Generate trading signals based on EMA (Exponential Moving Average) crossovers.
        
        This strategy generates:
        - Buy signals when the short-term EMA crosses above the long-term EMA
        - Sell signals when the short-term EMA crosses below the long-term EMA
        
        Parameters
        ----------
        market_data : MarketData
            Market data with price information
        short_window : int, default 8
            Window length for the short-term/faster EMA
        long_window : int, default 20
            Window length for the long-term/slower EMA
        price_col : str, default 'close'
            Column name containing the price data
        position_size : float, default 1.0
            Size of position to take on each signal (1 = 100%)
        strategy_name : str, default "EMA_Crossover"
            Name of the strategy
            
        Returns
        -------
        StrategySignals
            DataFrame with strategy signals
        """
        # Input validation
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        
        if price_col not in market_data.df.columns:
            raise ValueError(f"price_col '{price_col}' not found in DataFrame")
        
        # Access the dataframe from the MarketData object
        df = market_data.df.copy()
        
        # Calculate the two EMAs
        ema_short_col = f'EMA{short_window}'
        ema_long_col = f'EMA{long_window}'
        
        df[ema_short_col] = df[price_col].ewm(span=short_window, adjust=False, min_periods=short_window).mean()
        df[ema_long_col] = df[price_col].ewm(span=long_window, adjust=False, min_periods=long_window).mean()
        
        # Initialize signals dataframe
        signals_df = pd.DataFrame(index=df.index)
        signals_df['datetime_est_10mins'] = df['datetime_est_10mins']
        signals_df['close'] = df[price_col]
        signals_df['signal'] = 0.0
        
        # Generate buy signals (short EMA crosses above long EMA)
        bull_crossover = (
            (df[ema_short_col] > df[ema_long_col]) & 
            (df[ema_short_col].shift(1) <= df[ema_long_col].shift(1))
        )
        signals_df.loc[bull_crossover, 'signal'] = 1.0
        
        # Generate sell signals (short EMA crosses below long EMA)
        bear_crossover = (
            (df[ema_short_col] < df[ema_long_col]) & 
            (df[ema_short_col].shift(1) >= df[ema_long_col].shift(1))
        )
        signals_df.loc[bear_crossover, 'signal'] = -1.0
        
        # Create position column - maintain position until next signal
        positions = []
        current_position = 0
        
        for signal in signals_df['signal'].values:
            if signal != 0:
                # Scale the position by position_size while maintaining signal direction
                current_position = signal * position_size
            positions.append(current_position)
        
        signals_df['position'] = positions
        signals_df['position'] = signals_df['position'].shift(1)
        # Remove rows with NaN values from EMA calculation
        signals_df = signals_df.dropna()
        # Add strategy name
        signals_df['strategy_name'] = f"{strategy_name}_{short_window}_{long_window}"
        
        return StrategySignals(df=signals_df)
    
    @staticmethod
    def bollinger_bands_strategy(
        market_data: MarketData,
        window: int = 20,
        num_std: float = 2.0,
        position_size: float = 1.0,
        strategy_name: str = "BollingerBands",
        price_col: str = "close"
    ) -> StrategySignals:
        """
        Generate trading signals based on Bollinger Bands.
        
        Parameters
        ----------
        market_data : MarketData
            Market data containing price information
        window : int, default 20
            Window length for moving average calculation
        num_std : float, default 2.0
            Number of standard deviations for the bands
        position_size : float, default 1.0
            Size of position to take on each signal
        strategy_name : str, default "BollingerBands"
            Name of the strategy
        price_col : str, default "close"
            Column name for price data
            
        Returns
        -------
        StrategySignals
            DataFrame containing the generated signals
        """
        # Access the dataframe from the MarketData object
        df = market_data.df.copy()
        
        # Calculate Bollinger Bands
        df['MA'] = df[price_col].rolling(window=window).mean()
        df['BB_std'] = df[price_col].rolling(window=window).std()
        df['Upper_Band'] = df['MA'] + (df['BB_std'] * num_std)
        df['Lower_Band'] = df['MA'] - (df['BB_std'] * num_std)
        
        # Initialize signals dataframe
        signals_df = pd.DataFrame(index=df.index)
        signals_df['datetime_est_10mins'] = df['datetime_est_10mins']
        signals_df['close'] = df[price_col]
        signals_df['signal'] = 0.0
        
        # Generate buy signals (price crosses below lower band)
        buy_signals = (
            (df[price_col] < df['Lower_Band']) & 
            (df[price_col].shift(1) >= df['Lower_Band'].shift(1))
        )
        signals_df.loc[buy_signals, 'signal'] = 1.0
        
        # Generate sell signals (price crosses above upper band)
        sell_signals = (
            (df[price_col] > df['Upper_Band']) & 
            (df[price_col].shift(1) <= df['Upper_Band'].shift(1))
        )
        signals_df.loc[sell_signals, 'signal'] = -1.0
        
        # Create position column - maintain position until next signal
        positions = []
        current_position = 0
        
        for signal in signals_df['signal'].values:
            if signal != 0:
                # Scale the position by position_size while maintaining signal direction
                current_position = signal * position_size
            positions.append(current_position)
        
        signals_df['position'] = positions
        signals_df['position'] = signals_df['position'].shift(1)
        # Remove rows with NaN values from calculations
        signals_df = signals_df.dropna()
        # Add strategy name
        signals_df['strategy_name'] = f"{strategy_name}_{window}_{num_std}"
        
        return StrategySignals(df=signals_df)
    
    @staticmethod
    def oscillator_strategy(
        market_data: MarketData,
        window: int = 96,
        trigger: float = 0.0091,
        price_col: str = 'close',
        oscillator_col: str = 'oscillator',
        position_size: float = 1.0,
        strategy_name: str = "Oscillator_Strategy"
    ) -> StrategySignals:
        """
        Generate trading signals based on an oscillator crossing above/below trigger levels.
        
        This strategy generates:
        - Buy signals when oscillator crosses above trigger level
        - Sell signals when oscillator crosses below negative trigger level
        
        Parameters
        ----------
        market_data : MarketData
            Market data with price information
        window : int, default 96
            Window length for the oscillator calculation
        trigger : float, default 0.0091
            Trigger level for generating signals
        price_col : str, default 'close'
            Column name containing the price data
        oscillator_col : str, default 'oscillator'
            Column name containing the oscillator values
        position_size : float, default 1.0
            Size of position to take on each signal (1 = 100%)
        strategy_name : str, default "Oscillator_Strategy"
            Name of the strategy
            
        Returns
        -------
        StrategySignals
            DataFrame with strategy signals
        """
        # Input validation
        if not 0 < position_size <= 1:
            raise ValueError("position_size must be between 0 and 1")
            
        if oscillator_col not in market_data.df.columns:
            raise ValueError(f"oscillator_col '{oscillator_col}' not found in DataFrame")
        
        # Access the dataframe from the MarketData object
        df = market_data.df.copy()
        
        # Calculate the oscillator
        df['momentum'] = df[oscillator_col].rolling(window=5).mean() / df[oscillator_col].rolling(window=window).mean() - 1
        
        # Initialize signals dataframe
        signals_df = pd.DataFrame(index=df.index)
        signals_df['datetime_est_10mins'] = df['datetime_est_10mins']
        signals_df['close'] = df[price_col]
        signals_df['signal'] = 0.0
        
        # Generate signals based on oscillator crossing trigger levels
        signals_df.loc[df['momentum'] > trigger, 'signal'] = 1.0
        signals_df.loc[df['momentum'] < -trigger, 'signal'] = -1.0
        
        # Create position column - maintain position until next signal
        signals_df['position'] = signals_df['signal'].replace(to_replace=0, method='ffill')
        signals_df['position'] = signals_df['position'].shift(1)
        # Remove rows with NaN values
        signals_df = signals_df.dropna()
        signals_df['position'] = signals_df['position'] * position_size
        
        # Add strategy name
        signals_df['strategy_name'] = f"{strategy_name}_{window}_{trigger}"
        
        return StrategySignals(df=signals_df)
