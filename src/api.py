"""
Flask API - NFL Prediction Model
REST API for accessing NFL game predictions
"""

import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from typing import Dict, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_config, load_metrics, setup_logging,
    format_prediction_output, get_current_season, get_current_week
)
from src.predict import NFLPredictor


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
config = load_config()
logger = setup_logging(config.get('logging', {}).get('level', 'INFO'))

# Initialize predictor
predictor = NFLPredictor()

# API configuration
API_HOST = config['api']['host']
API_PORT = config['api']['port']
API_DEBUG = config['api']['debug']
API_KEY_REQUIRED = config['api'].get('api_key_required', False)
API_KEY = os.getenv('NFL_API_KEY', 'your-secret-api-key')


def verify_api_key() -> bool:
    """Verify API key if required"""
    if not API_KEY_REQUIRED:
        return True
    
    api_key = request.headers.get('X-API-Key')
    return api_key == API_KEY


def error_response(message: str, status_code: int = 400) -> tuple:
    """Create error response"""
    return jsonify({'error': message}), status_code


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'name': 'NFL Prediction Model API',
        'version': '1.0.0',
        'description': 'REST API for NFL game predictions',
        'endpoints': {
            '/': 'API information (this page)',
            '/health': 'Health check',
            '/predict': 'Get game predictions',
            '/predict/matchup': 'Get prediction for specific matchup',
            '/metrics': 'Get model performance metrics',
            '/teams': 'Get list of NFL teams'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'nfl-prediction-api',
        'timestamp': pd.Timestamp.now().isoformat()
    })


@app.route('/predict', methods=['GET'])
def get_predictions():
    """
    Get predictions for a specific week
    
    Query Parameters:
        - season (int, optional): Season year (default: current season)
        - week (int, optional): Week number (default: current week)
    
    Returns:
        JSON with predictions for all games in the specified week
    """
    if not verify_api_key():
        return error_response('Invalid or missing API key', 401)
    
    try:
        # Get query parameters
        season = request.args.get('season', type=int)
        week = request.args.get('week', type=int)
        
        # Default to current season/week if not provided
        if season is None:
            season = get_current_season()
        if week is None:
            week = get_current_week(season)
        
        # Validate parameters
        if season < 2010 or season > get_current_season() + 1:
            return error_response(f'Invalid season: {season}', 400)
        
        if week < 1 or week > 18:
            return error_response(f'Invalid week: {week}', 400)
        
        # Generate predictions
        logger.info(f"API request: predictions for season {season}, week {week}")
        predictions = predictor.predict(season, week)
        
        if predictions.empty:
            return jsonify({
                'season': season,
                'week': week,
                'predictions': [],
                'message': 'No games found for this week'
            })
        
        # Format output
        formatted_predictions = format_prediction_output(predictions)
        
        return jsonify({
            'season': season,
            'week': week,
            'total_games': len(formatted_predictions),
            'predictions': formatted_predictions
        })
    
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {e}")
        return error_response(f'Error generating predictions: {str(e)}', 500)


@app.route('/predict/matchup', methods=['GET'])
def get_matchup_prediction():
    """
    Get prediction for a specific matchup
    
    Query Parameters:
        - home (str, required): Home team abbreviation
        - away (str, required): Away team abbreviation
        - season (int, optional): Season year (default: current season)
        - week (int, optional): Week number (default: current week)
    
    Returns:
        JSON with prediction for the specified matchup
    """
    if not verify_api_key():
        return error_response('Invalid or missing API key', 401)
    
    try:
        # Get query parameters
        home_team = request.args.get('home', type=str)
        away_team = request.args.get('away', type=str)
        season = request.args.get('season', type=int)
        week = request.args.get('week', type=int)
        
        # Validate required parameters
        if not home_team or not away_team:
            return error_response('Missing required parameters: home and away', 400)
        
        # Convert to uppercase
        home_team = home_team.upper()
        away_team = away_team.upper()
        
        # Default to current season/week if not provided
        if season is None:
            season = get_current_season()
        if week is None:
            week = get_current_week(season)
        
        # Get prediction
        logger.info(f"API request: matchup prediction for {away_team} @ {home_team}")
        result = predictor.predict_matchup(home_team, away_team, season, week)
        
        if not result:
            return error_response(f'Matchup not found: {away_team} @ {home_team}', 404)
        
        # Format output
        return jsonify({
            'season': season,
            'week': week,
            'matchup': f'{away_team} @ {home_team}',
            'prediction': {
                'home_team': home_team,
                'away_team': away_team,
                'home_win_prob': round(float(result['home_win_prob']), 3),
                'away_win_prob': round(float(result['away_win_prob']), 3),
                'confidence': round(float(result['confidence']), 3),
                'predicted_winner': result['predicted_winner']
            }
        })
    
    except Exception as e:
        logger.error(f"Error in /predict/matchup endpoint: {e}")
        return error_response(f'Error generating prediction: {str(e)}', 500)


@app.route('/metrics', methods=['GET'])
def get_model_metrics():
    """
    Get model performance metrics
    
    Returns:
        JSON with model accuracy, ROC-AUC, log loss, and other metrics
    """
    try:
        metrics = load_metrics()
        
        if not metrics:
            return jsonify({
                'message': 'No metrics available. Train a model first.'
            })
        
        return jsonify({
            'model_metrics': {
                'accuracy': round(metrics.get('accuracy', 0), 4),
                'roc_auc': round(metrics.get('roc_auc', 0), 4),
                'log_loss': round(metrics.get('log_loss', 0), 4),
                'precision': round(metrics.get('precision', 0), 4),
                'recall': round(metrics.get('recall', 0), 4),
                'f1_score': round(metrics.get('f1_score', 0), 4)
            },
            'cross_validation': {
                'cv_mean': round(metrics.get('cv_mean', 0), 4),
                'cv_std': round(metrics.get('cv_std', 0), 4)
            } if 'cv_mean' in metrics else None,
            'last_updated': metrics.get('last_updated', 'Unknown'),
            'model_name': metrics.get('model_name', 'Unknown')
        })
    
    except Exception as e:
        logger.error(f"Error in /metrics endpoint: {e}")
        return error_response(f'Error loading metrics: {str(e)}', 500)


@app.route('/teams', methods=['GET'])
def get_teams():
    """
    Get list of NFL teams
    
    Returns:
        JSON with list of team abbreviations
    """
    # NFL team abbreviations (2024)
    teams = [
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
        'LA', 'LAC', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
        'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    ]
    
    return jsonify({
        'teams': teams,
        'total': len(teams)
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return error_response('Endpoint not found', 404)


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return error_response('Internal server error', 500)


def main():
    """Run the Flask API server"""
    logger.info("=" * 60)
    logger.info("NFL Prediction Model - API Server")
    logger.info("=" * 60)
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    
    if API_KEY_REQUIRED:
        logger.info("API key authentication is ENABLED")
    else:
        logger.warning("API key authentication is DISABLED")
    
    logger.info("=" * 60)
    
    app.run(
        host=API_HOST,
        port=API_PORT,
        debug=API_DEBUG
    )


if __name__ == "__main__":
    main()

