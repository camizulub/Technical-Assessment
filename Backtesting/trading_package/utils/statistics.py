"""
Statistical utility functions for trading calculations.
"""

import numpy as np
import pandas as pd


class StatisticsUtils:
    """
    Utility class for statistical calculations used in trading analysis.
    """
    
    @staticmethod
    def geometric_mean(returns: pd.Series) -> float:
        """
        Calculate the geometric mean of a series of returns.
        
        Parameters
        ----------
        returns : pd.Series
            Series of returns (as decimal, e.g., 0.01 for 1%)
            
        Returns
        -------
        float
            Geometric mean return
        """
        if len(returns) == 0:
            return 0
        
        # Convert returns to gross returns (1 + r)
        gross_returns = 1 + returns
        
        # Calculate geometric mean
        return np.exp(np.mean(np.log(gross_returns))) - 1
