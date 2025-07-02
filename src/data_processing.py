import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, 
                                 FunctionTransformer)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xverse.transformer import MonotonicBinning
from sklearn.compose import make_column_selector

class WOETransformer(BaseEstimator, TransformerMixin):
    """Custom Weight of Evidence transformer"""
    def __init__(self, target=None):
        self.target = target
        self.woe_dict = {}
        
    def fit(self, X, y=None):
        if y is None and self.target is None:
            raise ValueError("WOE transformation requires target variable")
        
        target = y if y is not None else self.target
        
        for col in X.columns:
            if X[col].nunique() > 1:  # Only calculate WOE for varying features
                woe_df = pd.DataFrame({'x': X[col], 'y': target})
                woe_table = woe_df.groupby('x')['y'].agg(['count', 'mean'])
                woe_table['woe'] = np.log((1 - woe_table['mean']) / woe_table['mean'])
                self.woe_dict[col] = woe_table['woe'].to_dict()
        return self
        
    def transform(self, X):
        X_woe = X.copy()
        for col in self.woe_dict:
            X_woe[col] = X[col].map(self.woe_dict[col]).fillna(0)
        return X_woe

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts datetime features from TransactionStartTime"""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionWeekday'] = X['TransactionStartTime'].dt.weekday
        return X[['TransactionHour', 'TransactionDay', 
                 'TransactionMonth', 'TransactionWeekday']]

class RFMAggregator(BaseEstimator, TransformerMixin):
    """Calculates RFM features at customer level and generates target variable"""
    def __init__(self, target_threshold=0.5):
        self.target_threshold = target_threshold
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        # Calculate RFM features
        rfm = X.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (pd.Timestamp.now() - x.max()).days,
            'TransactionId': 'count',
            'Amount': ['sum', 'mean'],
            'Default': 'mean'  # Assuming you have a Default column for supervised learning
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary_Sum', 'Monetary_Mean', 'DefaultRate']
        
        # Create target variable
        rfm['is_high_risk'] = (rfm['DefaultRate'] > self.target_threshold).astype(int)
        
        return rfm.reset_index()
def create_full_pipeline():
    """Create the complete feature engineering pipeline"""
    # Numerical features
    numeric_features = ['Amount', 'Value']
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features
    categorical_features = ['ProductCategory', 'ChannelId']
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    
    return pipeline

