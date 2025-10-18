"""
Data Fetching Script - NFL Prediction Model
Fetches historical NFL game data from nflfastR/NFLVerse and other sources
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Optional
import nfl_data_py as nfl
from tqdm import tqdm
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, create_directories, setup_logging


class NFLDataFetcher:
    """Fetch and store NFL data from various sources"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data fetcher with configuration"""
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config.get('logging', {}).get('level', 'INFO'))
        create_directories(self.config)
        
        self.raw_dir = self.config['data']['raw_dir']
        self.start_season = self.config['data']['start_season']
        self.end_season = self.config['data']['end_season']
        
        self.logger.info(f"Initialized NFLDataFetcher for seasons {self.start_season}-{self.end_season}")
    
    def fetch_schedules(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Fetch game schedules with results
        
        Args:
            seasons: List of seasons to fetch. If None, uses config range.
            
        Returns:
            DataFrame with game schedules and results
        """
        if seasons is None:
            seasons = list(range(self.start_season, self.end_season + 1))
        
        self.logger.info(f"Fetching schedules for seasons: {seasons}")
        
        try:
            schedules = nfl.import_schedules(seasons)
            
            # Save to CSV
            output_path = os.path.join(self.raw_dir, "schedules.csv")
            schedules.to_csv(output_path, index=False)
            self.logger.info(f"Saved schedules to {output_path} ({len(schedules)} games)")
            
            return schedules
        
        except Exception as e:
            self.logger.error(f"Error fetching schedules: {e}")
            raise
    
    def fetch_team_stats(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Fetch weekly team statistics
        
        Args:
            seasons: List of seasons to fetch. If None, uses config range.
            
        Returns:
            DataFrame with team statistics by week
        """
        if seasons is None:
            seasons = list(range(self.start_season, self.end_season + 1))
        
        self.logger.info(f"Fetching team stats for seasons: {seasons}")
        
        try:
            weekly_stats = nfl.import_weekly_data(seasons, downcast=True)
            
            # Aggregate to team level
            team_stats = self._aggregate_team_stats(weekly_stats)
            
            # Save to CSV
            output_path = os.path.join(self.raw_dir, "team_stats.csv")
            team_stats.to_csv(output_path, index=False)
            self.logger.info(f"Saved team stats to {output_path}")
            
            return team_stats
        
        except Exception as e:
            self.logger.error(f"Error fetching team stats: {e}")
            raise
    
    def _aggregate_team_stats(self, weekly_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player stats to team level"""
        
        # Select relevant columns for aggregation
        agg_cols = {
            'passing_yards': 'sum',
            'passing_tds': 'sum',
            'interceptions': 'sum',
            'rushing_yards': 'sum',
            'rushing_tds': 'sum',
            'receiving_yards': 'sum',
            'receiving_tds': 'sum',
            'sacks': 'sum',
            'fumbles_lost': 'sum',
            'fantasy_points': 'sum'
        }
        
        # Filter to only include columns that exist
        agg_cols_filtered = {k: v for k, v in agg_cols.items() if k in weekly_data.columns}
        
        # Group by team, season, week
        group_cols = ['season', 'week', 'recent_team']
        
        team_stats = weekly_data.groupby(group_cols).agg(agg_cols_filtered).reset_index()
        team_stats.rename(columns={'recent_team': 'team'}, inplace=True)
        
        return team_stats
    
    def fetch_pbp_stats(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Fetch play-by-play data and calculate advanced metrics
        
        Args:
            seasons: List of seasons to fetch. If None, uses config range.
            
        Returns:
            DataFrame with play-by-play statistics
        """
        if seasons is None:
            seasons = list(range(self.start_season, self.end_season + 1))
        
        self.logger.info(f"Fetching play-by-play data for seasons: {seasons}")
        
        try:
            pbp_data = []
            
            for season in tqdm(seasons, desc="Fetching PBP data"):
                try:
                    season_pbp = nfl.import_pbp_data([season], downcast=True)
                    pbp_data.append(season_pbp)
                except Exception as e:
                    self.logger.warning(f"Error fetching PBP for season {season}: {e}")
            
            if pbp_data:
                pbp_df = pd.concat(pbp_data, ignore_index=True)
                
                # Save to parquet for efficiency
                output_path = os.path.join(self.raw_dir, "pbp_data.parquet")
                pbp_df.to_parquet(output_path, index=False)
                self.logger.info(f"Saved PBP data to {output_path}")
                
                return pbp_df
            else:
                self.logger.warning("No PBP data fetched")
                return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Error fetching PBP stats: {e}")
            raise
    
    def fetch_team_descriptions(self) -> pd.DataFrame:
        """Fetch team information and metadata"""
        self.logger.info("Fetching team descriptions")
        
        try:
            teams = nfl.import_team_desc()
            
            # Save to CSV
            output_path = os.path.join(self.raw_dir, "teams.csv")
            teams.to_csv(output_path, index=False)
            self.logger.info(f"Saved team descriptions to {output_path}")
            
            return teams
        
        except Exception as e:
            self.logger.error(f"Error fetching team descriptions: {e}")
            raise
    
    def fetch_rosters(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Fetch player rosters (optional - not all nfl-data-py versions support this)
        
        Args:
            seasons: List of seasons to fetch. If None, uses config range.
            
        Returns:
            DataFrame with player roster information
        """
        if seasons is None:
            seasons = list(range(self.start_season, self.end_season + 1))
        
        self.logger.info(f"Fetching rosters for seasons: {seasons}")
        
        try:
            # Check if import_rosters exists
            if not hasattr(nfl, 'import_rosters'):
                self.logger.warning("import_rosters not available in this nfl-data-py version. Skipping.")
                return pd.DataFrame()
            
            rosters = nfl.import_rosters(seasons)
            
            # Save to CSV
            output_path = os.path.join(self.raw_dir, "rosters.csv")
            rosters.to_csv(output_path, index=False)
            self.logger.info(f"Saved rosters to {output_path}")
            
            return rosters
        
        except Exception as e:
            self.logger.warning(f"Could not fetch rosters: {e}. Continuing without roster data.")
            return pd.DataFrame()
    
    def fetch_all_data(self) -> None:
        """Fetch all required data sources"""
        self.logger.info("Starting comprehensive data fetch")
        
        try:
            # Fetch schedules (most important)
            self.logger.info("=" * 60)
            self.logger.info("Step 1/5: Fetching schedules...")
            self.fetch_schedules()
            
            # Fetch team descriptions
            self.logger.info("=" * 60)
            self.logger.info("Step 2/5: Fetching team descriptions...")
            self.fetch_team_descriptions()
            
            # Fetch team stats
            self.logger.info("=" * 60)
            self.logger.info("Step 3/5: Fetching team stats...")
            self.fetch_team_stats()
            
            # Fetch rosters (optional)
            self.logger.info("=" * 60)
            self.logger.info("Step 4/5: Fetching rosters (optional)...")
            try:
                self.fetch_rosters()
            except Exception as e:
                self.logger.warning(f"Skipping rosters: {e}")
            
            # Fetch play-by-play (this may take a while)
            self.logger.info("=" * 60)
            self.logger.info("Step 5/5: Fetching play-by-play data (this may take several minutes)...")
            try:
                self.fetch_pbp_stats()
            except Exception as e:
                self.logger.warning(f"Could not fetch PBP data: {e}. Continuing without it.")
            
            self.logger.info("=" * 60)
            self.logger.info("✅ Data fetch complete! All data saved to data/raw/")
        
        except Exception as e:
            self.logger.error(f"Error during comprehensive data fetch: {e}")
            raise


def main():
    """Main execution function"""
    print("=" * 60)
    print("NFL Prediction Model - Data Fetching")
    print("=" * 60)
    
    try:
        fetcher = NFLDataFetcher()
        fetcher.fetch_all_data()
        
        print("\n✅ Data fetching completed successfully!")
        print(f"Data saved to: {fetcher.raw_dir}/")
        
    except Exception as e:
        print(f"\n❌ Error during data fetching: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

