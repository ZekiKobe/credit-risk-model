import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, 
                                 FunctionTransformer)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xverse.transformer import MonotonicBinning

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
    """Creates RFM features at customer level"""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        rfm = X.groupby('AccountId').agg({
            'TransactionStartTime': lambda x: (x.max() - x.min()).days,
            'TransactionId': 'count',
            'Amount': ['sum', 'mean', 'std']
        })
        rfm.columns = ['Recency', 'Frequency', 
                      'Monetary_Sum', 'Monetary_Mean', 'Monetary_Std']
        return rfm.reset_index()

def build_feature_pipeline():
    """Main pipeline construction"""
    
    # Temporal features pipeline
    temporal_pipe = Pipeline([
        ('extractor', TemporalFeatureExtractor()),
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    # RFM features pipeline
    rfm_pipe = Pipeline([
        ('aggregator', RFMAggregator()),
        ('scaler', StandardScaler())
    ])
    
    # Categorical processing
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Numerical processing
    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Column mappings
    categorical_features = ['ProductCategory', 'ChannelId', 'CountryCode']
    numerical_features = ['Amount', 'Value']
    
    # Main preprocessor
    preprocessor = ColumnTransformer([
        ('temporal', temporal_pipe, ['TransactionStartTime']),
        ('categorical', categorical_pipe, categorical_features),
        ('numerical', numerical_pipe, numerical_features),
        ('rfm', rfm_pipe, ['AccountId', 'TransactionStartTime', 
                          'TransactionId', 'Amount'])
    ], remainder='drop')
    
    # Full pipeline with custom WOE
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('monotonic_bin', MonotonicBinning())
    ])
    
    return full_pipeline