import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, precision_score, 
                           recall_score, f1_score, accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskProxyGenerator:
    """Generates proxy target variable based on RFM analysis"""
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
        ('preprocessor', preprocessor),
        ('feature_selector', SelectKBest(score_func=f_classif, k='all'))
    ])
    
    return pipeline

class CreditRiskTrainer:
    def __init__(self, transaction_data_path: str, rfm_data_path: str = None, experiment_name: str = "credit_risk"):
        self.transaction_data_path = transaction_data_path
        self.rfm_data_path = rfm_data_path
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the raw data"""
        try:
            # Load transaction data
            df = pd.read_csv(
                self.transaction_data_path,
                parse_dates=['TransactionStartTime']
            )
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
            
            # Load RFM data if available
            if self.rfm_data_path:
                rfm_df = pd.read_csv(self.rfm_data_path)
                # Ensure we don't create duplicate columns
                rfm_df = rfm_df.rename(columns={'AccountId': 'CustomerId'})
                df = pd.merge(
                    df,
                    rfm_df,
                    on='CustomerId',
                    how='left'
                )
            
            # Validate required columns
            required_cols = {
                'CustomerId', 'TransactionStartTime', 'Amount',
                'ProductCategory', 'ChannelId', 'Value'
            }
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
                
            # Clean data
            df['CustomerId'] = df['CustomerId'].astype(str)
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            
            # Drop rows with missing critical values
            initial_count = len(df)
            df = df.dropna(subset=['CustomerId', 'TransactionStartTime', 'Amount'])
            if len(df) < initial_count:
                logger.warning(f"Dropped {initial_count - len(df)} rows with missing values")
            
            return df
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data and prepare features and target"""
        try:
            df = self.load_data()
            logger.info(f"Data loaded successfully with shape: {df.shape}")
            
            # Generate target (already at customer level)
            target_engineer = RiskProxyGenerator()
            y_df = target_engineer.generate_proxy_target(df)
            
            # Feature engineering - first aggregate to customer level
            customer_features = df.groupby('CustomerId').agg({
                'Amount': ['mean', 'sum', 'std'],
                'Value': ['mean', 'sum', 'std'],
                'ProductCategory': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'missing',
                'ChannelId': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'missing',
                'TransactionStartTime': lambda x: (x.max() - x.min()).days
            })
            
            # Flatten multi-index columns
            customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns]
            customer_features = customer_features.reset_index()
            
            # Apply preprocessing pipeline
            feature_pipeline = create_full_pipeline()
            
            # Prepare features for pipeline (only include columns that exist)
            features_to_use = ['Amount_mean', 'Amount_sum', 'Amount_std', 
                             'Value_mean', 'Value_sum', 'Value_std',
                             'ProductCategory', 'ChannelId']
            features_to_use = [col for col in features_to_use if col in customer_features.columns]
            
            X = feature_pipeline.fit_transform(customer_features[features_to_use])
            
            # Get feature names
            feature_names = feature_pipeline.get_feature_names_out()
            
            # Create final DataFrame
            X_df = pd.DataFrame(X, columns=feature_names)
            X_df['CustomerId'] = customer_features['CustomerId']
            
            # Merge with target
            merged = pd.merge(
                X_df,
                y_df[['CustomerId', 'is_high_risk']],
                on='CustomerId',
                how='inner'
            )
            
            logger.info(f"Final features shape: {merged.shape}")
            logger.info(f"Class distribution:\n{merged['is_high_risk'].value_counts(normalize=True)}")
            
            return merged.drop(['is_high_risk', 'CustomerId'], axis=1), merged['is_high_risk']
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
    def train_models(self) -> str:
        """Train and evaluate multiple models"""
        try:
            X, y = self.load_and_prepare_data()
            
            # Data validation
            assert not X.isnull().any().any(), "Missing values found in features"
            assert len(np.unique(y)) > 1, "Target has only one class"
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Calculate class weights for imbalanced data
            class_weights = len(y_train[y_train==0])/len(y_train[y_train==1])
            
            models = {
                "logistic_regression": LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=42
                ),
                "random_forest": RandomForestClassifier(
                    class_weight='balanced',
                    random_state=42
                ),
                "xgboost": XGBClassifier(
                    scale_pos_weight=class_weights,
                    random_state=42,
                    eval_metric='auc',
                    use_label_encoder=False
                )
            }
            
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                with mlflow.start_run(run_name=name):
                    # Cross-validation
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=StratifiedKFold(5),
                        scoring='roc_auc'
                    )
                    mlflow.log_metric("cv_mean_auc", cv_scores.mean())
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:,1]
                    
                    # Calculate metrics
                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred),
                        "recall": recall_score(y_test, y_pred),
                        "f1": f1_score(y_test, y_pred),
                        "roc_auc": roc_auc_score(y_test, y_proba)
                    }
                    
                    # Log metrics and model
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(model, name)
                    
                    # Track best model
                    if metrics["roc_auc"] > best_score:
                        best_score = metrics["roc_auc"]
                        best_model = name
                    
                    logger.info(f"{name} - ROC AUC: {metrics['roc_auc']:.4f}")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        trainer = CreditRiskTrainer(
            transaction_data_path="../data/processed/processed_data.csv",
            rfm_data_path="../data/processed/rfm_data.csv"
        )
        
        # First validate data loading
        test_df = trainer.load_data()
        logger.info("Data validation successful. Sample data:")
        logger.info(test_df.head())
        
        # Proceed with training
        logger.info("Starting model training...")
        best_model = trainer.train_models()
        logger.info(f"Best performing model: {best_model}")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise