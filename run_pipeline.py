#!/usr/bin/env python3
"""
Complete Pipeline Runner - NFL Prediction Model
Runs the entire pipeline from data fetching to predictions
"""

import os
import sys
import argparse
from datetime import datetime


def run_fetch_data():
    """Run data fetching"""
    print("\n" + "=" * 60)
    print("STEP 1: Fetching Data")
    print("=" * 60)
    from src.fetch_data import main as fetch_main
    fetch_main()


def run_build_features():
    """Run feature engineering"""
    print("\n" + "=" * 60)
    print("STEP 2: Building Features")
    print("=" * 60)
    from src.build_features import main as features_main
    features_main()


def run_train_model():
    """Run model training"""
    print("\n" + "=" * 60)
    print("STEP 3: Training Model")
    print("=" * 60)
    from src.train import main as train_main
    train_main()


def run_generate_predictions(season=None, week=None):
    """Run predictions"""
    print("\n" + "=" * 60)
    print("STEP 4: Generating Predictions")
    print("=" * 60)
    
    # Temporarily modify sys.argv for predict script
    original_argv = sys.argv.copy()
    sys.argv = ['predict.py']
    
    if season:
        sys.argv.extend(['--season', str(season)])
    if week:
        sys.argv.extend(['--week', str(week)])
    
    from src.predict import main as predict_main
    predict_main()
    
    sys.argv = original_argv


def run_full_pipeline(season=None, week=None):
    """Run the complete pipeline"""
    start_time = datetime.now()
    
    print("=" * 60)
    print("NFL PREDICTION MODEL - FULL PIPELINE")
    print("=" * 60)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Step 1: Fetch data
        run_fetch_data()
        
        # Step 2: Build features
        run_build_features()
        
        # Step 3: Train model
        run_train_model()
        
        # Step 4: Generate predictions
        run_generate_predictions(season, week)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Total time: {duration}")
        print("=" * 60)
        
        print("\n📊 Next steps:")
        print("  1. View predictions in: predictions/")
        print("  2. Launch dashboard: streamlit run src/dashboard.py")
        print("  3. Start API server: python src/api.py")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run the NFL Prediction Model pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (fetch, features, train, predict)
  python run_pipeline.py --full
  
  # Run individual steps
  python run_pipeline.py --fetch
  python run_pipeline.py --features
  python run_pipeline.py --train
  python run_pipeline.py --predict
  
  # Generate predictions for specific week
  python run_pipeline.py --predict --season 2024 --week 5
        """
    )
    
    # Pipeline steps
    parser.add_argument('--full', action='store_true',
                       help='Run the complete pipeline (fetch, features, train, predict)')
    parser.add_argument('--fetch', action='store_true',
                       help='Fetch data from sources')
    parser.add_argument('--features', action='store_true',
                       help='Build features from raw data')
    parser.add_argument('--train', action='store_true',
                       help='Train prediction models')
    parser.add_argument('--predict', action='store_true',
                       help='Generate predictions')
    
    # Prediction parameters
    parser.add_argument('--season', type=int, default=None,
                       help='Season year for predictions')
    parser.add_argument('--week', type=int, default=None,
                       help='Week number for predictions')
    
    args = parser.parse_args()
    
    # Check if any action was specified
    if not any([args.full, args.fetch, args.features, args.train, args.predict]):
        parser.print_help()
        sys.exit(0)
    
    # Run requested pipeline steps
    if args.full:
        run_full_pipeline(args.season, args.week)
    else:
        if args.fetch:
            run_fetch_data()
        
        if args.features:
            run_build_features()
        
        if args.train:
            run_train_model()
        
        if args.predict:
            run_generate_predictions(args.season, args.week)


if __name__ == "__main__":
    main()

