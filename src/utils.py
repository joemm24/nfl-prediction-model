"""
Utility functions for NFL Prediction Model
"""

import os
import yaml
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories if they don't exist"""
    dirs = [
        config['data']['raw_dir'],
        config['data']['processed_dir'],
        config['data']['features_dir'],
        config['model']['save_dir'],
        config['predictions']['output_dir']
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def save_model(model: Any, model_name: str, save_dir: str = "models") -> str:
    """Save a trained model to disk"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)
    joblib.dump(model, filepath)
    
    # Also save as latest
    latest_path = os.path.join(save_dir, f"{model_name}_latest.pkl")
    joblib.dump(model, latest_path)
    
    return filepath


def load_model(model_name: str, save_dir: str = "models", use_latest: bool = True) -> Any:
    """Load a trained model from disk"""
    if use_latest:
        filepath = os.path.join(save_dir, f"{model_name}_latest.pkl")
    else:
        filepath = os.path.join(save_dir, f"{model_name}.pkl")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    return joblib.load(filepath)


def save_predictions(predictions: pd.DataFrame, output_dir: str = "predictions", 
                     formats: List[str] = ["json", "csv"]) -> Dict[str, str]:
    """Save predictions in multiple formats"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}
    
    for fmt in formats:
        if fmt == "json":
            filename = f"predictions_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            predictions.to_json(filepath, orient='records', indent=2)
            saved_files['json'] = filepath
            
        elif fmt == "csv":
            filename = f"predictions_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            predictions.to_csv(filepath, index=False)
            saved_files['csv'] = filepath
    
    # Also save as latest
    if "json" in formats:
        latest_json = os.path.join(output_dir, "predictions_latest.json")
        predictions.to_json(latest_json, orient='records', indent=2)
    
    if "csv" in formats:
        latest_csv = os.path.join(output_dir, "predictions_latest.csv")
        predictions.to_csv(latest_csv, index=False)
    
    return saved_files


def save_metrics(metrics: Dict[str, float], filepath: str = "models/metrics.json") -> None:
    """Save model performance metrics"""
    metrics['last_updated'] = datetime.now().isoformat()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: str = "models/metrics.json") -> Dict[str, Any]:
    """Load model performance metrics"""
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_rolling_stats(df: pd.DataFrame, group_col: str, stat_cols: List[str], 
                           window: int = 4, min_periods: int = 1) -> pd.DataFrame:
    """Calculate rolling averages for specified columns"""
    result_df = df.copy()
    
    for col in stat_cols:
        if col in df.columns:
            rolling_col_name = f"{col}_rolling_{window}"
            result_df[rolling_col_name] = (
                result_df.groupby(group_col)[col]
                .transform(lambda x: x.rolling(window=window, min_periods=min_periods).mean())
            )
    
    return result_df


def calculate_team_differentials(df: pd.DataFrame, team_cols: List[str]) -> pd.DataFrame:
    """Calculate differentials between home and away team stats"""
    result_df = df.copy()
    
    for col in team_cols:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        diff_col = f"{col}_differential"
        
        if home_col in df.columns and away_col in df.columns:
            result_df[diff_col] = result_df[home_col] - result_df[away_col]
    
    return result_df


def get_current_season() -> int:
    """Get the current NFL season year"""
    now = datetime.now()
    # NFL season typically starts in September
    if now.month >= 9:
        return now.year
    else:
        return now.year - 1


def get_current_week(season: Optional[int] = None) -> int:
    """
    Get the current NFL week
    Note: This is a simplified version. In production, you'd want to 
    fetch this from an API or maintain a schedule database.
    """
    if season is None:
        season = get_current_season()
    
    # Simplified: assume season starts first week of September
    now = datetime.now()
    if now.year == season:
        # Rough estimate based on weeks since September 1st
        season_start = datetime(season, 9, 1)
        if now >= season_start:
            weeks_elapsed = (now - season_start).days // 7
            return min(weeks_elapsed + 1, 18)  # Max 18 weeks regular season
    
    return 1


def validate_data(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
    """Validate that required columns exist in dataframe"""
    missing_cols = [col for col in required_cols if col not in df.columns]
    is_valid = len(missing_cols) == 0
    return is_valid, missing_cols


def clean_team_names(df: pd.DataFrame, team_cols: List[str]) -> pd.DataFrame:
    """Standardize team abbreviations"""
    result_df = df.copy()
    
    # Mapping for team name variations
    team_mapping = {
        'STL': 'LA',   # St. Louis Rams -> LA Rams
        'SD': 'LAC',   # San Diego Chargers -> LA Chargers
        'OAK': 'LV',   # Oakland Raiders -> Las Vegas Raiders
    }
    
    for col in team_cols:
        if col in result_df.columns:
            result_df[col] = result_df[col].replace(team_mapping)
    
    return result_df


def get_feature_importance(model: Any, feature_names: List[str], 
                          top_n: int = 20) -> pd.DataFrame:
    """Extract and rank feature importance from trained model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def format_prediction_output(predictions: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format predictions for API/dashboard consumption"""
    formatted = []
    
    for _, row in predictions.iterrows():
        formatted.append({
            'game_id': row.get('game_id', ''),
            'season': int(row.get('season', 0)),
            'week': int(row.get('week', 0)),
            'home_team': row.get('home_team', ''),
            'away_team': row.get('away_team', ''),
            'home_win_prob': round(float(row.get('home_win_prob', 0)), 3),
            'away_win_prob': round(float(row.get('away_win_prob', 0)), 3),
            'confidence': round(float(row.get('confidence', 0)), 3),
            'predicted_winner': row.get('predicted_winner', ''),
            'gameday': row.get('gameday', '')
        })
    
    return formatted


def calculate_confidence(probabilities: np.ndarray) -> np.ndarray:
    """
    Calculate confidence score from win probabilities
    Confidence is the absolute difference from 0.5 (range: 0 to 0.5)
    """
    return np.abs(probabilities - 0.5)

