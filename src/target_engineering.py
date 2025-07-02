# src/target_engineering.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RiskProxyGenerator:
    def __init__(self, recency_threshold=90, frequency_threshold=5, monetary_threshold=1000):
        self.recency_threshold = recency_threshold
        self.frequency_threshold = frequency_threshold
        self.monetary_threshold = monetary_threshold
        
    def generate_proxy_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate risk proxy target based on RFM analysis"""
        try:
            # Validate input columns
            required_cols = {'CustomerId', 'TransactionStartTime', 'Amount'}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(f"Missing required columns: {missing}")

            # Convert to datetime with proper timezone handling
            df['TransactionStartTime'] = pd.to_datetime(
                df['TransactionStartTime'],
                utc=True,  # Important for timezone-aware timestamps
                errors='coerce'  # Convert invalid dates to NaT
            )
            
            # Check for failed conversions
            if df['TransactionStartTime'].isna().any():
                bad_dates = df[df['TransactionStartTime'].isna()]['TransactionStartTime'].head()
                logger.warning(f"Failed to parse {df['TransactionStartTime'].isna().sum()} dates. Examples:\n{bad_dates}")
                df = df.dropna(subset=['TransactionStartTime'])
            
            # Calculate RFM metrics
            rfm = df.groupby('CustomerId').agg({
                'TransactionStartTime': lambda x: (pd.Timestamp.now(tz='UTC') - x.max()).days,
                'Amount': ['count', 'sum']
            })
            
            # Flatten multi-index columns
            rfm.columns = ['Recency', 'Frequency', 'Monetary']
            
            # Create risk flags
            rfm['is_high_risk'] = np.where(
                (rfm['Recency'] > self.recency_threshold) |
                (rfm['Frequency'] < self.frequency_threshold) |
                (rfm['Monetary'] > self.monetary_threshold),
                1, 0
            )
            
            return rfm.reset_index()
            
        except Exception as e:
            logger.error(f"Target generation failed: {str(e)}")
            raise
