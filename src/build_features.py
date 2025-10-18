"""
Feature Engineering Pipeline - NFL Prediction Model
Transforms raw data into features for machine learning models
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_config, create_directories, setup_logging,
    calculate_rolling_stats, clean_team_names
)


class NFLFeatureBuilder:
    """Build ML features from raw NFL data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize feature builder with configuration"""
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config.get('logging', {}).get('level', 'INFO'))
        create_directories(self.config)
        
        self.raw_dir = self.config['data']['raw_dir']
        self.processed_dir = self.config['data']['processed_dir']
        self.features_dir = self.config['data']['features_dir']
        self.rolling_window = self.config['features']['rolling_window']
        self.min_games = self.config['features']['min_games']
        
        self.logger.info("Initialized NFLFeatureBuilder")
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw data files"""
        self.logger.info("Loading raw data files...")
        
        data = {}
        
        # Load schedules (required)
        schedules_path = os.path.join(self.raw_dir, "schedules.csv")
        if os.path.exists(schedules_path):
            data['schedules'] = pd.read_csv(schedules_path)
            self.logger.info(f"Loaded schedules: {len(data['schedules'])} games")
        else:
            raise FileNotFoundError(f"Schedules file not found: {schedules_path}")
        
        # Load team stats (optional but recommended)
        team_stats_path = os.path.join(self.raw_dir, "team_stats.csv")
        if os.path.exists(team_stats_path):
            data['team_stats'] = pd.read_csv(team_stats_path)
            self.logger.info(f"Loaded team stats: {len(data['team_stats'])} records")
        
        # Load teams (optional)
        teams_path = os.path.join(self.raw_dir, "teams.csv")
        if os.path.exists(teams_path):
            data['teams'] = pd.read_csv(teams_path)
            self.logger.info(f"Loaded teams: {len(data['teams'])} teams")
        
        # Load PBP data if available (for advanced metrics)
        pbp_path = os.path.join(self.raw_dir, "pbp_data.parquet")
        if os.path.exists(pbp_path):
            data['pbp'] = pd.read_parquet(pbp_path)
            self.logger.info(f"Loaded PBP data: {len(data['pbp'])} plays")
        
        return data
    
    def prepare_game_results(self, schedules: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare game results with clean team names and outcomes
        
        Args:
            schedules: Raw schedule data
            
        Returns:
            Cleaned schedule dataframe
        """
        self.logger.info("Preparing game results...")
        
        df = schedules.copy()
        
        # Filter to regular season games only
        df = df[df['game_type'] == 'REG'].copy()
        
        # Clean team names
        df = clean_team_names(df, ['home_team', 'away_team'])
        
        # Create target variable: home team win
        df['home_team_win'] = (df['home_score'] > df['away_score']).astype(int)
        
        # Calculate point differential
        df['point_differential'] = df['home_score'] - df['away_score']
        
        # Calculate total points
        df['total_points'] = df['home_score'] + df['away_score']
        
        # Sort by date
        df = df.sort_values(['season', 'week', 'gameday']).reset_index(drop=True)
        
        self.logger.info(f"Prepared {len(df)} regular season games")
        
        return df
    
    def calculate_team_season_stats(self, schedules: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate season-level team statistics
        
        Args:
            schedules: Cleaned schedule data
            
        Returns:
            DataFrame with team season stats
        """
        self.logger.info("Calculating team season statistics...")
        
        # Process home games
        home_stats = schedules.groupby(['season', 'home_team']).agg({
            'home_score': ['mean', 'sum', 'std'],
            'away_score': ['mean'],
            'home_team_win': ['sum', 'mean']
        }).reset_index()
        
        home_stats.columns = ['season', 'team', 'points_scored_mean', 'points_scored_total', 
                             'points_scored_std', 'points_allowed_mean', 
                             'wins', 'win_rate']
        home_stats['games_played'] = schedules.groupby(['season', 'home_team']).size().values
        
        # Process away games
        away_stats = schedules.groupby(['season', 'away_team']).agg({
            'away_score': ['mean', 'sum', 'std'],
            'home_score': ['mean'],
            'home_team_win': ['count']
        }).reset_index()
        
        away_stats.columns = ['season', 'team', 'away_points_scored_mean', 
                             'away_points_scored_total', 'away_points_scored_std',
                             'away_points_allowed_mean', 'away_games_played']
        
        # Calculate away wins
        away_wins = schedules[schedules['home_team_win'] == 0].groupby(['season', 'away_team']).size()
        away_stats = away_stats.merge(
            away_wins.reset_index(name='away_wins'),
            on=['season', 'team'],
            how='left'
        )
        away_stats['away_wins'] = away_stats['away_wins'].fillna(0)
        
        # Merge home and away stats
        team_stats = home_stats.merge(
            away_stats,
            on=['season', 'team'],
            how='outer'
        ).fillna(0)
        
        # Calculate combined statistics
        team_stats['total_wins'] = team_stats['wins'] + team_stats['away_wins']
        team_stats['total_games'] = team_stats['games_played'] + team_stats['away_games_played']
        team_stats['overall_win_rate'] = team_stats['total_wins'] / team_stats['total_games'].replace(0, 1)
        
        self.logger.info(f"Calculated stats for {len(team_stats)} team-seasons")
        
        return team_stats
    
    def calculate_rolling_team_stats(self, schedules: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling averages for team performance
        
        Args:
            schedules: Cleaned schedule data
            
        Returns:
            DataFrame with rolling statistics
        """
        self.logger.info(f"Calculating rolling stats (window={self.rolling_window})...")
        
        # Create a combined dataset for all team games
        home_games = schedules[['season', 'week', 'gameday', 'home_team', 
                                'home_score', 'away_score', 'home_team_win']].copy()
        home_games.rename(columns={
            'home_team': 'team',
            'home_score': 'points_scored',
            'away_score': 'points_allowed',
            'home_team_win': 'win'
        }, inplace=True)
        home_games['is_home'] = 1
        
        away_games = schedules[['season', 'week', 'gameday', 'away_team', 
                                'away_score', 'home_score', 'home_team_win']].copy()
        away_games.rename(columns={
            'away_team': 'team',
            'away_score': 'points_scored',
            'home_score': 'points_allowed'
        }, inplace=True)
        away_games['win'] = (1 - away_games['home_team_win']).astype(int)
        away_games['is_home'] = 0
        away_games = away_games.drop('home_team_win', axis=1)
        
        # Combine all games
        all_games = pd.concat([home_games, away_games], ignore_index=True)
        all_games = all_games.sort_values(['season', 'week', 'team']).reset_index(drop=True)
        
        # Calculate point differential
        all_games['point_differential'] = all_games['points_scored'] - all_games['points_allowed']
        
        # Group by team and season, then calculate rolling stats
        rolling_cols = ['points_scored', 'points_allowed', 'point_differential', 'win']
        
        for col in rolling_cols:
            all_games[f'{col}_rolling'] = (
                all_games.groupby(['season', 'team'])[col]
                .transform(lambda x: x.shift(1).rolling(window=self.rolling_window, 
                                                         min_periods=self.min_games).mean())
            )
        
        # Calculate form (last 3 games)
        all_games['recent_form'] = (
            all_games.groupby(['season', 'team'])['win']
            .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        )
        
        self.logger.info(f"Calculated rolling stats for {len(all_games)} team-game records")
        
        return all_games
    
    def calculate_advanced_metrics(self, pbp_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced metrics from play-by-play data (EPA, success rate, etc.)
        
        Args:
            pbp_data: Play-by-play data
            
        Returns:
            DataFrame with advanced metrics by team, season, week
        """
        self.logger.info("Calculating advanced metrics from PBP data...")
        
        # Filter to relevant plays
        pbp = pbp_data[pbp_data['play_type'].isin(['pass', 'run'])].copy()
        
        # Calculate offensive EPA
        off_epa = pbp.groupby(['season', 'week', 'posteam']).agg({
            'epa': 'mean',
            'success': 'mean',
            'yards_gained': 'mean'
        }).reset_index()
        
        off_epa.rename(columns={
            'posteam': 'team',
            'epa': 'epa_per_play',
            'success': 'success_rate',
            'yards_gained': 'yards_per_play'
        }, inplace=True)
        
        # Calculate defensive EPA (opponent perspective)
        def_epa = pbp.groupby(['season', 'week', 'defteam']).agg({
            'epa': 'mean',
            'success': 'mean'
        }).reset_index()
        
        def_epa.rename(columns={
            'defteam': 'team',
            'epa': 'epa_allowed_per_play',
            'success': 'success_rate_allowed'
        }, inplace=True)
        
        # Merge offensive and defensive metrics
        advanced_metrics = off_epa.merge(
            def_epa,
            on=['season', 'week', 'team'],
            how='outer'
        ).fillna(0)
        
        self.logger.info(f"Calculated advanced metrics for {len(advanced_metrics)} team-week records")
        
        return advanced_metrics
    
    def merge_game_features(self, schedules: pd.DataFrame, rolling_stats: pd.DataFrame,
                           advanced_metrics: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge all features into a single dataset aligned with games
        
        Args:
            schedules: Game schedules
            rolling_stats: Rolling team statistics
            advanced_metrics: Advanced metrics (optional)
            
        Returns:
            Complete feature dataset
        """
        self.logger.info("Merging features into game-level dataset...")
        
        df = schedules.copy()
        
        # Merge home team rolling stats
        df = df.merge(
            rolling_stats[['season', 'week', 'team', 'points_scored_rolling', 
                          'points_allowed_rolling', 'point_differential_rolling',
                          'win_rolling', 'recent_form']],
            left_on=['season', 'week', 'home_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_home')
        ).drop('team', axis=1)
        
        # Rename home team columns
        df.rename(columns={
            'points_scored_rolling': 'home_points_scored_rolling',
            'points_allowed_rolling': 'home_points_allowed_rolling',
            'point_differential_rolling': 'home_point_diff_rolling',
            'win_rolling': 'home_win_rate_rolling',
            'recent_form': 'home_recent_form'
        }, inplace=True)
        
        # Merge away team rolling stats
        df = df.merge(
            rolling_stats[['season', 'week', 'team', 'points_scored_rolling', 
                          'points_allowed_rolling', 'point_differential_rolling',
                          'win_rolling', 'recent_form']],
            left_on=['season', 'week', 'away_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_away')
        ).drop('team', axis=1)
        
        # Rename away team columns
        df.rename(columns={
            'points_scored_rolling': 'away_points_scored_rolling',
            'points_allowed_rolling': 'away_points_allowed_rolling',
            'point_differential_rolling': 'away_point_diff_rolling',
            'win_rolling': 'away_win_rate_rolling',
            'recent_form': 'away_recent_form'
        }, inplace=True)
        
        # Merge advanced metrics if available
        if advanced_metrics is not None:
            # Home team advanced metrics
            df = df.merge(
                advanced_metrics,
                left_on=['season', 'week', 'home_team'],
                right_on=['season', 'week', 'team'],
                how='left',
                suffixes=('', '_home')
            ).drop('team', axis=1, errors='ignore')
            
            # Rename home columns
            for col in ['epa_per_play', 'success_rate', 'yards_per_play', 
                       'epa_allowed_per_play', 'success_rate_allowed']:
                if col in df.columns:
                    df.rename(columns={col: f'home_{col}'}, inplace=True)
            
            # Away team advanced metrics
            df = df.merge(
                advanced_metrics,
                left_on=['season', 'week', 'away_team'],
                right_on=['season', 'week', 'team'],
                how='left',
                suffixes=('', '_away')
            ).drop('team', axis=1, errors='ignore')
            
            # Rename away columns
            for col in ['epa_per_play', 'success_rate', 'yards_per_play', 
                       'epa_allowed_per_play', 'success_rate_allowed']:
                if col in df.columns:
                    df.rename(columns={col: f'away_{col}'}, inplace=True)
        
        # Calculate differentials
        diff_cols = [
            'points_scored_rolling', 'points_allowed_rolling', 'point_diff_rolling',
            'win_rate_rolling', 'recent_form'
        ]
        
        for col in diff_cols:
            home_col = f'home_{col}'
            away_col = f'away_{col}'
            if home_col in df.columns and away_col in df.columns:
                df[f'{col}_differential'] = df[home_col] - df[away_col]
        
        # Add additional features
        df['is_divisional_game'] = (df['home_team'].str[:2] == df['away_team'].str[:2]).astype(int)
        
        self.logger.info(f"Merged features for {len(df)} games with {len(df.columns)} features")
        
        return df
    
    def build_features(self) -> pd.DataFrame:
        """Main pipeline to build all features"""
        self.logger.info("=" * 60)
        self.logger.info("Starting feature engineering pipeline")
        self.logger.info("=" * 60)
        
        # Load raw data
        data = self.load_raw_data()
        
        # Prepare game results
        schedules = self.prepare_game_results(data['schedules'])
        
        # Calculate rolling stats
        rolling_stats = self.calculate_rolling_team_stats(schedules)
        
        # Calculate advanced metrics if PBP data available
        advanced_metrics = None
        if 'pbp' in data:
            try:
                advanced_metrics = self.calculate_advanced_metrics(data['pbp'])
            except Exception as e:
                self.logger.warning(f"Could not calculate advanced metrics: {e}")
        
        # Merge all features
        features = self.merge_game_features(schedules, rolling_stats, advanced_metrics)
        
        # Filter to complete games only (for training)
        features_complete = features[features['home_score'].notna()].copy()
        
        # Save processed data
        features_path = os.path.join(self.features_dir, "features.csv")
        features_complete.to_csv(features_path, index=False)
        self.logger.info(f"Saved features to {features_path}")
        
        # Also save all features including future games (for prediction)
        features_all_path = os.path.join(self.features_dir, "features_all.csv")
        features.to_csv(features_all_path, index=False)
        self.logger.info(f"Saved all features to {features_all_path}")
        
        self.logger.info("=" * 60)
        self.logger.info(f"✅ Feature engineering complete!")
        self.logger.info(f"Features shape: {features_complete.shape}")
        self.logger.info(f"Null values: {features_complete.isnull().sum().sum()}")
        self.logger.info("=" * 60)
        
        return features_complete


def main():
    """Main execution function"""
    print("=" * 60)
    print("NFL Prediction Model - Feature Engineering")
    print("=" * 60)
    
    try:
        builder = NFLFeatureBuilder()
        features = builder.build_features()
        
        print(f"\n✅ Feature engineering completed successfully!")
        print(f"Features shape: {features.shape}")
        print(f"Saved to: {builder.features_dir}/")
        
    except Exception as e:
        print(f"\n❌ Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

