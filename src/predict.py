"""
Prediction Script - NFL Prediction Model
Generate predictions for upcoming NFL games
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_config, create_directories, setup_logging,
    load_model, save_predictions, format_prediction_output,
    calculate_confidence, get_current_season, get_current_week
)


class NFLPredictor:
    """Generate predictions for NFL games"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize predictor with configuration"""
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config.get('logging', {}).get('level', 'INFO'))
        create_directories(self.config)
        
        self.features_dir = self.config['data']['features_dir']
        self.model_dir = self.config['model']['save_dir']
        self.predictions_dir = self.config['predictions']['output_dir']
        self.export_formats = self.config['predictions']['export_formats']
        
        self.logger.info("Initialized NFLPredictor")
    
    def load_best_model(self) -> Any:
        """Load the best trained model"""
        try:
            model = load_model('best_model', self.model_dir)
            self.logger.info("Loaded best model successfully")
            return model
        except FileNotFoundError:
            self.logger.error("Best model not found. Please train a model first.")
            raise
    
    def load_features_for_prediction(self, season: Optional[int] = None,
                                    week: Optional[int] = None) -> pd.DataFrame:
        """
        Load features for games to predict
        
        Args:
            season: Season year (defaults to current)
            week: Week number (defaults to current)
            
        Returns:
            DataFrame with features for upcoming games
        """
        features_path = os.path.join(self.features_dir, "features_all.csv")
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"Features file not found: {features_path}. "
                "Please run build_features.py first."
            )
        
        df = pd.read_csv(features_path)
        
        # Default to current season and week
        if season is None:
            season = get_current_season()
        if week is None:
            week = get_current_week(season)
        
        self.logger.info(f"Loaded features for season {season}, week {week}")
        
        # Filter to specified season and week
        df_filtered = df[(df['season'] == season) & (df['week'] == week)].copy()
        
        if len(df_filtered) == 0:
            self.logger.warning(f"No games found for season {season}, week {week}")
        else:
            self.logger.info(f"Found {len(df_filtered)} games for prediction")
        
        return df_filtered
    
    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for prediction
        
        Args:
            df: Raw feature dataframe
            
        Returns:
            Tuple of (metadata, features)
        """
        # Columns to keep as metadata
        metadata_cols = [
            'game_id', 'season', 'week', 'gameday', 'weekday', 'gametime',
            'home_team', 'away_team', 'home_score', 'away_score',
            'location', 'stadium', 'roof', 'surface'
        ]
        
        # Select metadata columns that exist
        metadata_cols_existing = [col for col in metadata_cols if col in df.columns]
        metadata = df[metadata_cols_existing].copy()
        
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
        
        # Extract features
        X = df[feature_cols].copy()
        
        # Fill any NaN values
        X = X.fillna(0)
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        self.logger.info(f"Prepared features: {X.shape}")
        
        return metadata, X
    
    def generate_predictions(self, model: Any, metadata: pd.DataFrame,
                           X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for games
        
        Args:
            model: Trained model
            metadata: Game metadata
            X: Features
            
        Returns:
            DataFrame with predictions
        """
        self.logger.info("Generating predictions...")
        
        # Get prediction probabilities
        probabilities = model.predict_proba(X)
        home_win_prob = probabilities[:, 1]
        away_win_prob = probabilities[:, 0]
        
        # Calculate confidence
        confidence = calculate_confidence(home_win_prob)
        
        # Determine predicted winner
        predicted_winner = np.where(home_win_prob > 0.5, 
                                    metadata['home_team'].values, 
                                    metadata['away_team'].values)
        
        # Create predictions dataframe
        predictions = metadata.copy()
        predictions['home_win_prob'] = home_win_prob
        predictions['away_win_prob'] = away_win_prob
        predictions['confidence'] = confidence
        predictions['predicted_winner'] = predicted_winner
        predictions['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def predict(self, season: Optional[int] = None, 
                week: Optional[int] = None) -> pd.DataFrame:
        """
        Main prediction pipeline
        
        Args:
            season: Season year (defaults to current)
            week: Week number (defaults to current)
            
        Returns:
            DataFrame with predictions
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting prediction pipeline")
        self.logger.info("=" * 60)
        
        # Load model
        model = self.load_best_model()
        
        # Load features
        df = self.load_features_for_prediction(season, week)
        
        if len(df) == 0:
            self.logger.warning("No games to predict")
            return pd.DataFrame()
        
        # Prepare features
        metadata, X = self.prepare_features(df)
        
        # Generate predictions
        predictions = self.generate_predictions(model, metadata, X)
        
        # Save predictions
        saved_files = save_predictions(predictions, self.predictions_dir, self.export_formats)
        
        for fmt, path in saved_files.items():
            self.logger.info(f"Saved predictions ({fmt}): {path}")
        
        self.logger.info("=" * 60)
        self.logger.info("✅ Prediction pipeline complete!")
        self.logger.info("=" * 60)
        
        return predictions
    
    def predict_matchup(self, home_team: str, away_team: str,
                       season: Optional[int] = None,
                       week: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict a specific matchup
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number
            
        Returns:
            Dictionary with prediction details
        """
        if season is None:
            season = get_current_season()
        if week is None:
            week = get_current_week(season)
        
        # Load all predictions for the week
        predictions = self.predict(season, week)
        
        # Filter to specific matchup
        matchup = predictions[
            (predictions['home_team'] == home_team) & 
            (predictions['away_team'] == away_team)
        ]
        
        if len(matchup) == 0:
            self.logger.warning(f"Matchup not found: {away_team} @ {home_team}")
            return {}
        
        # Format output
        result = matchup.iloc[0].to_dict()
        
        return result
    
    def display_predictions(self, predictions: pd.DataFrame) -> None:
        """
        Display predictions in a readable format
        
        Args:
            predictions: Predictions dataframe
        """
        if len(predictions) == 0:
            print("\nNo predictions available.")
            return
        
        print("\n" + "=" * 80)
        print(f"NFL GAME PREDICTIONS - Week {predictions['week'].iloc[0]}, {predictions['season'].iloc[0]}")
        print("=" * 80)
        
        for idx, row in predictions.iterrows():
            home_prob = row['home_win_prob']
            away_prob = row['away_win_prob']
            confidence = row['confidence']
            
            print(f"\n{row['away_team']} @ {row['home_team']}")
            if 'gameday' in row and pd.notna(row['gameday']):
                print(f"  Date: {row['gameday']}")
            
            print(f"  Home Win Probability: {home_prob:.1%}")
            print(f"  Away Win Probability: {away_prob:.1%}")
            print(f"  Predicted Winner: {row['predicted_winner']}")
            print(f"  Confidence: {confidence:.1%}")
            
            # Visual probability bar
            bar_length = 40
            home_bar = int(home_prob * bar_length)
            away_bar = bar_length - home_bar
            
            print(f"  [{row['home_team']}] {'█' * home_bar}{' ' * away_bar} [{row['away_team']}]")
        
        print("\n" + "=" * 80)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate NFL game predictions')
    parser.add_argument('--season', type=int, default=None, 
                       help='Season year (default: current season)')
    parser.add_argument('--week', type=int, default=None,
                       help='Week number (default: current week)')
    parser.add_argument('--home', type=str, default=None,
                       help='Home team abbreviation (for specific matchup)')
    parser.add_argument('--away', type=str, default=None,
                       help='Away team abbreviation (for specific matchup)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NFL Prediction Model - Predictions")
    print("=" * 60)
    
    try:
        predictor = NFLPredictor()
        
        if args.home and args.away:
            # Predict specific matchup
            result = predictor.predict_matchup(args.home, args.away, args.season, args.week)
            if result:
                print(f"\nPrediction for {args.away} @ {args.home}:")
                print(f"  Home Win Probability: {result['home_win_prob']:.1%}")
                print(f"  Away Win Probability: {result['away_win_prob']:.1%}")
                print(f"  Predicted Winner: {result['predicted_winner']}")
                print(f"  Confidence: {result['confidence']:.1%}")
        else:
            # Predict all games for week
            predictions = predictor.predict(args.season, args.week)
            predictor.display_predictions(predictions)
        
        print("\n✅ Predictions generated successfully!")
        print(f"Saved to: {predictor.predictions_dir}/")
        
    except Exception as e:
        print(f"\n❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

