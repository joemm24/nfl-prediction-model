"""
Model Training Script - NFL Prediction Model
Train and evaluate machine learning models for NFL game prediction
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    # XGBoost may fail to load if OpenMP is not installed
    XGBOOST_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError, Exception):
    # LightGBM may fail to load if OpenMP is not installed
    LIGHTGBM_AVAILABLE = False
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_config, create_directories, setup_logging,
    save_model, save_metrics, get_feature_importance
)


class NFLModelTrainer:
    """Train and evaluate NFL prediction models"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model trainer with configuration"""
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config.get('logging', {}).get('level', 'INFO'))
        create_directories(self.config)
        
        self.features_dir = self.config['data']['features_dir']
        self.model_dir = self.config['model']['save_dir']
        self.target = self.config['model']['target']
        self.test_size = self.config['model']['test_size']
        self.random_state = self.config['model']['random_state']
        self.cv_folds = self.config['model']['cv_folds']
        
        self.models = {}
        self.results = {}
        
        self.logger.info("Initialized NFLModelTrainer")
    
    def load_features(self) -> pd.DataFrame:
        """Load engineered features"""
        features_path = os.path.join(self.features_dir, "features.csv")
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        df = pd.read_csv(features_path)
        self.logger.info(f"Loaded features: {df.shape}")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features and target for modeling
        
        Args:
            df: Feature dataframe
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        self.logger.info("Preparing data for modeling...")
        
        # Select feature columns (exclude metadata and target)
        exclude_cols = [
            'game_id', 'gameday', 'weekday', 'gametime', 'season', 'week',
            'home_team', 'away_team', 'home_score', 'away_score',
            'home_team_win', 'point_differential', 'total_points',
            'location', 'result', 'overtime', 'old_game_id', 'gsis',
            'nfl_detail_id', 'pfr', 'pff', 'espn', 'stadium_id', 'stadium',
            'roof', 'surface', 'temp', 'wind', 'away_rest', 'home_rest',
            'away_moneyline', 'home_moneyline', 'spread_line', 'away_spread_odds',
            'home_spread_odds', 'total_line', 'under_odds', 'over_odds',
            'div_game', 'home_coach', 'away_coach', 'referee',
            'away_qb_id', 'home_qb_id', 'away_qb_name', 'home_qb_name'
        ]
        
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Ensure target exists
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in data")
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[self.target].copy()
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        self.logger.info(f"Prepared data: X shape={X.shape}, y shape={y.shape}")
        self.logger.info(f"Number of features: {len(feature_cols)}")
        self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train Logistic Regression model"""
        self.logger.info("Training Logistic Regression...")
        
        params = self.config['hyperparameters']['logistic_regression']
        
        model = LogisticRegression(**params, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        self.logger.info("✓ Logistic Regression trained")
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train Random Forest model"""
        self.logger.info("Training Random Forest...")
        
        params = self.config['hyperparameters']['random_forest']
        
        model = RandomForestClassifier(**params, random_state=self.random_state, n_jobs=1)
        model.fit(X_train, y_train)
        
        self.logger.info("✓ Random Forest trained")
        return model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install libomp: brew install libomp")
        
        self.logger.info("Training XGBoost...")
        
        params = self.config['hyperparameters']['xgboost']
        
        model = xgb.XGBClassifier(
            **params,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        self.logger.info("✓ XGBoost trained")
        return model
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Log metrics
        self.logger.info(f"\n{model_name} Performance:")
        self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        self.logger.info(f"  Log Loss:  {metrics['log_loss']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.logger.info(f"  Confusion Matrix:\n{cm}")
        
        return metrics
    
    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                            model_name: str) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            model_name: Name of the model
            
        Returns:
            Dictionary of CV metrics
        """
        self.logger.info(f"Cross-validating {model_name}...")
        
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, 
                                   scoring='roc_auc', n_jobs=1)  # Serial to avoid sandbox issues
        
        cv_metrics = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        self.logger.info(f"  CV ROC-AUC: {cv_metrics['cv_mean']:.4f} (+/- {cv_metrics['cv_std']:.4f})")
        
        return cv_metrics
    
    def train_all_models(self) -> Dict[str, Any]:
        """Train all configured models and select the best one"""
        self.logger.info("=" * 60)
        self.logger.info("Starting model training pipeline")
        self.logger.info("=" * 60)
        
        # Load and prepare data
        df = self.load_features()
        X, y, feature_names = self.prepare_data(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Store feature names
        self.feature_names = feature_names
        
        # Train models
        model_configs = self.config['model']['models']
        
        for model_name in model_configs:
            self.logger.info("=" * 60)
            
            if model_name == 'logistic_regression':
                model = self.train_logistic_regression(X_train, y_train)
            elif model_name == 'random_forest':
                model = self.train_random_forest(X_train, y_train)
            elif model_name == 'xgboost':
                model = self.train_xgboost(X_train, y_train)
            else:
                self.logger.warning(f"Unknown model: {model_name}")
                continue
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            # Cross-validation
            cv_metrics = self.cross_validate_model(model, X, y, model_name)
            metrics.update(cv_metrics)
            
            # Store model and results
            self.models[model_name] = model
            self.results[model_name] = metrics
            
            # Save model
            save_model(model, model_name, self.model_dir)
            self.logger.info(f"✓ Saved {model_name} model")
        
        # Select best model
        best_model_name = self._select_best_model()
        best_model = self.models[best_model_name]
        best_metrics = self.results[best_model_name]
        
        # Save best model separately
        save_model(best_model, 'best_model', self.model_dir)
        
        # Save all metrics
        save_metrics(best_metrics, os.path.join(self.model_dir, 'metrics.json'))
        
        # Save detailed results
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(os.path.join(self.model_dir, 'model_comparison.csv'))
        
        # Feature importance for best model
        if best_model_name in ['random_forest', 'xgboost']:
            importance_df = get_feature_importance(best_model, feature_names, top_n=20)
            importance_df.to_csv(os.path.join(self.model_dir, 'feature_importance.csv'), index=False)
            
            self.logger.info("\nTop 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.logger.info("=" * 60)
        self.logger.info(f"✅ Training complete! Best model: {best_model_name}")
        self.logger.info(f"Best model accuracy: {best_metrics['accuracy']:.4f}")
        self.logger.info(f"Best model ROC-AUC: {best_metrics['roc_auc']:.4f}")
        self.logger.info("=" * 60)
        
        return {
            'best_model_name': best_model_name,
            'best_model': best_model,
            'all_models': self.models,
            'results': self.results
        }
    
    def _select_best_model(self) -> str:
        """Select the best model based on primary metric"""
        primary_metric = self.config['metrics']['primary']
        
        best_model_name = max(self.results, key=lambda k: self.results[k][primary_metric])
        
        self.logger.info(f"\nBest model selected: {best_model_name}")
        self.logger.info(f"  {primary_metric}: {self.results[best_model_name][primary_metric]:.4f}")
        
        return best_model_name


def main():
    """Main execution function"""
    print("=" * 60)
    print("NFL Prediction Model - Training")
    print("=" * 60)
    
    try:
        trainer = NFLModelTrainer()
        results = trainer.train_all_models()
        
        print(f"\n✅ Model training completed successfully!")
        print(f"Best model: {results['best_model_name']}")
        print(f"Models saved to: {trainer.model_dir}/")
        
    except Exception as e:
        print(f"\n❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

