"""
Data models for trading package using Pydantic for validation.
"""

from pydantic import BaseModel, field_validator, ConfigDict
import pandas as pd
from typing import Dict, Any


class MarketData(BaseModel):
    """Market data model with validation for required price data."""
    df: pd.DataFrame
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("df")
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that the DataFrame has the required columns."""
        required_columns = {"datetime_est_10mins", "close"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Ensure datetime column is properly formatted
        if not isinstance(df["datetime_est_10mins"].iloc[0], pd.Timestamp):
            raise ValueError("datetime_est_10mins column must be datetime type")
        
        # Check for duplicate datetime values
        if df["datetime_est_10mins"].duplicated().any():
            raise ValueError("datetime_est_10mins column contains duplicate values")
        
        return df


class StrategySignals(BaseModel):
    """Strategy signals model with validation for signal data."""
    df: pd.DataFrame
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("df")
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that the DataFrame has the required columns and proper signal values."""
        required_columns = {"datetime_est_10mins", "signal", "position", "strategy_name", "close"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Validate signal column is float containing -1, 0, or 1
        if not df["signal"].dtype.kind == 'f':
            raise ValueError("Signal column must be float type")
            
        if not (df["signal"].isin([-1, 0, 1])).all():
            raise ValueError("Signal values must be either -1, 0, or 1")
        
        # Validate position column is float between -1 and 1
        if not df["position"].dtype.kind == 'f':
            raise ValueError("Position column must be float type")
            
        if not ((df["position"] >= -1) & (df["position"] <= 1)).all():
            raise ValueError("Position values must be between -1 and 1")
        # Validate strategy_name is a string and consistent throughout the DataFrame
        if not df["strategy_name"].dtype.kind == 'O':  # 'O' indicates object/string type
            raise ValueError("strategy_name column must be string type")
        
        if df["strategy_name"].nunique() > 1:
            raise ValueError("All rows in strategy_name column must have the same value")
            
        return df
    
    @property
    def strategy_name(self) -> str:
        """Get the strategy name from the DataFrame."""
        return self.df["strategy_name"].iloc[0]


class TradeData(BaseModel):
    """Trade data model with validation for trade information."""
    df: pd.DataFrame
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("df")
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that the DataFrame has the required columns for trade data."""
        required_columns = {"Size", "EntryBar", "ExitBar", "EntryPrice", "ExitPrice", 
                           "PnL", "ReturnPct", "EntryTime", "ExitTime"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        return df


class PerformanceMetrics(BaseModel):
    """Performance metrics model for strategy evaluation results."""
    metrics: Dict[str, Any]
    strategy_name: str
    
    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the metrics dictionary contains essential metrics."""
        essential_metrics = {"Sharpe Ratio", "Sortino Ratio", "Max. Drawdown [%]"}
        if not essential_metrics.issubset(metrics.keys()):
            raise ValueError(f"Metrics must contain at least: {essential_metrics}")
        return metrics 
