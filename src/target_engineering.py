import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

class RiskProxyGenerator:
    """
    Creates proxy target variable using RFM analysis and clustering
    """
    
    def __init__(self, snapshot_date=None, random_state=42):
        self.snapshot_date = pd.to_datetime(snapshot_date) if snapshot_date else None
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=random_state)
        self.rfm_pipeline = Pipeline([
            ('scaler', self.scaler),
            ('cluster', self.kmeans)
        ])
        self.high_risk_cluster = None
        self.cluster_stats = None
        self.logger = logging.getLogger(__name__)
        
    def calculate_rfm(self, df):
        """Calculate RFM metrics for each customer"""
        required_cols = ['CustomerId', 'TransactionStartTime', 'TransactionId', 'Amount']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataframe must contain {required_cols}")
            
        if not self.snapshot_date:
            self.snapshot_date = df['TransactionStartTime'].max()
            
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': lambda x: x[x > 0].sum()  # Only consider purchases
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary'
        })
        
        # Log transform with epsilon to handle zero
        rfm['Monetary'] = np.log1p(rfm['Monetary'])
        return rfm
    
    def identify_high_risk_cluster(self, rfm):
        """Determine which cluster represents high-risk customers"""
        self.cluster_stats = rfm.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).reset_index()
        
        # High risk = high recency, low frequency, low monetary
        self.cluster_stats['RiskScore'] = (
            self.cluster_stats['Recency'].rank(ascending=False) +
            self.cluster_stats['Frequency'].rank(ascending=True) +
            self.cluster_stats['Monetary'].rank(ascending=True)
        )
        
        self.high_risk_cluster = self.cluster_stats.loc[
            self.cluster_stats['RiskScore'].idxmin(), 'Cluster'
        ]
        self.logger.info(f"Identified high-risk cluster: {self.high_risk_cluster}")
        return self.high_risk_cluster
    
    def generate_proxy_target(self, df):
        """Main method to create is_high_risk column"""
        # Step 1: Calculate RFM metrics
        rfm = self.calculate_rfm(df)
        
        # Step 2: Cluster customers
        features = ['Recency', 'Frequency', 'Monetary']
        rfm['Cluster'] = self.rfm_pipeline.fit_predict(rfm[features])
        
        # Step 3: Identify high-risk cluster
        self.identify_high_risk_cluster(rfm)
        
        # Step 4: Create binary target
        rfm['is_high_risk'] = (rfm['Cluster'] == self.high_risk_cluster).astype(int)
        
        # Step 5: Merge back to original data
        result = df.merge(
            rfm[['is_high_risk']], 
            left_on='CustomerId', 
            right_index=True,
            how='left'
        )
        
        # Treat new customers as high risk
        result['is_high_risk'] = result['is_high_risk'].fillna(1)
        return result
    
    def get_cluster_profiles(self):
        """Returns cluster characteristics for analysis"""
        if self.cluster_stats is None:
            raise ValueError("Run generate_proxy_target() first")
        return self.cluster_stats.sort_values('RiskScore').to_dict('records')