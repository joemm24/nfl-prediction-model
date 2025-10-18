# 🏈 NFL Game Prediction Model

A machine learning-based system that forecasts NFL game outcomes using historical data, team performance, coaching history, and contextual factors.

## 🎯 Features

- **Data Ingestion**: Automated ETL pipeline from nflfastR/NFLVerse
- **Feature Engineering**: 40+ engineered features including rolling averages, home/away splits, and advanced metrics
- **Machine Learning**: Multiple models (Logistic Regression, Random Forest, XGBoost)
- **Interactive Dashboard**: Built with Streamlit for visualization and exploration
- **REST API**: Flask-based API for programmatic access
- **Weekly Updates**: Automated retraining and prediction generation

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nfl-prediction-model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Fetch Data
```bash
python src/fetch_data.py
```

#### 2. Build Features
```bash
python src/build_features.py
```

#### 3. Train Model
```bash
python src/train.py
```

#### 4. Generate Predictions
```bash
python src/predict.py
```

#### 5. Launch Dashboard
```bash
streamlit run src/dashboard.py
```

#### 6. Start API Server
```bash
python src/api.py
```

## 📊 Model Performance

Target Metrics:
- **Accuracy**: ≥ 65%
- **ROC-AUC**: ≥ 0.70
- **Log Loss**: Minimized for calibrated probabilities

## 🏗️ Project Structure

```
nfl-prediction-model/
├── data/
│   ├── raw/              # Raw data from sources
│   ├── processed/        # Cleaned and processed data
│   └── features/         # Engineered features
├── models/               # Saved model files
├── predictions/          # Weekly prediction outputs
├── src/
│   ├── fetch_data.py     # Data ingestion
│   ├── build_features.py # Feature engineering
│   ├── train.py          # Model training
│   ├── predict.py        # Generate predictions
│   ├── dashboard.py      # Streamlit dashboard
│   ├── api.py            # Flask API
│   └── utils.py          # Helper functions
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔌 API Endpoints

### GET /predict
Returns win probabilities for upcoming matchups.

**Query Parameters:**
- `week` (optional): Specific week number
- `season` (optional): Season year

**Response:**
```json
{
  "predictions": [
    {
      "game_id": "2024_01_KC_DET",
      "home_team": "DET",
      "away_team": "KC",
      "home_win_prob": 0.58,
      "away_win_prob": 0.42,
      "confidence": 0.16
    }
  ]
}
```

### GET /metrics
Returns model performance metrics.

**Response:**
```json
{
  "accuracy": 0.67,
  "roc_auc": 0.72,
  "log_loss": 0.61,
  "last_updated": "2024-10-18"
}
```

## 📈 Dashboard Features

- **Matchup Cards**: Visual display of game predictions with win probabilities
- **Team Filters**: Filter by specific teams or weeks
- **Performance Tracking**: Historical accuracy and calibration charts
- **Confidence Visualization**: Gauge charts showing prediction confidence

## 🛠️ Tech Stack

- **Language**: Python 3.11+
- **Data**: pandas, numpy, nfl-data-py
- **ML**: scikit-learn, XGBoost, LightGBM
- **Dashboard**: Streamlit
- **API**: Flask
- **Storage**: SQLite/CSV

## 📝 Future Enhancements

- Player-level statistics and injury data
- Vegas line integration for EV calculations
- Ensemble models combining ELO + ML
- Live in-game win probability updates
- User authentication and prediction history
- Automated accuracy summaries

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

