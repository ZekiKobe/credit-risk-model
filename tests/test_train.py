import pytest
from src.train import CreditRiskModelTrainer
import pandas as pd
import numpy as np
from unittest.mock import patch

class TestCreditRiskModelTrainer:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'Amount': [100, 200, 300, 400, 500],
            'Value': [1, 2, 3, 4, 5],
            'is_high_risk': [0, 1, 0, 1, 0],
            'ProductCategory': ['A', 'B', 'A', 'C', 'B']
        })
    
    def test_data_loading(self, sample_data):
        """Test data loading and preprocessing"""
        with patch('pandas.read_csv', return_value=sample_data):
            trainer = CreditRiskModelTrainer("dummy_path")
            processed_data = trainer.load_data()
            
            assert not processed_data.empty
            assert 'is_high_risk' in processed_data.columns
            
    def test_model_training(self, sample_data):
        """Test model training workflow"""
        with patch('pandas.read_csv', return_value=sample_data), \
             patch('mlflow.start_run'), \
             patch('sklearn.linear_model.LogisticRegression.fit') as mock_fit:
            
            trainer = CreditRiskModelTrainer("dummy_path")
            trainer.train_models()
            
            mock_fit.assert_called_once()
            
    def test_hyperparameter_tuning(self):
        """Test hyperparameter optimization"""
        with patch('hyperopt.fmin', return_value={'max_depth': 5, 'learning_rate': 0.1}):
            trainer = CreditRiskModelTrainer("dummy_path")
            trainer.hyperparameter_tuning()