# 🚀 Quick Start Guide

Get up and running with the NFL Prediction Model in just a few steps!

## Prerequisites

- Python 3.11 or higher
- 2GB+ free disk space (for data)
- Internet connection (for data fetching)

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd nfl-prediction-model

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

## Running the Model

### Full Pipeline (First Time)

Run the complete pipeline to fetch data, build features, train models, and generate predictions:

```bash
python run_pipeline.py --full
```

This will take approximately 15-30 minutes depending on your internet connection and CPU.

### Individual Steps

You can also run each step separately:

```bash
# 1. Fetch data from nflfastR/NFLVerse
python run_pipeline.py --fetch

# 2. Build features from raw data
python run_pipeline.py --features

# 3. Train prediction models
python run_pipeline.py --train

# 4. Generate predictions for current week
python run_pipeline.py --predict

# Or predict specific week:
python run_pipeline.py --predict --season 2024 --week 5
```

## Using the Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run src/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Dashboard Features

- 📊 View predictions for all games in a week
- 🎯 Filter by team or confidence level
- 📈 Visualize win probabilities
- 💾 Download predictions as CSV

## Using the API

Start the Flask API server:

```bash
python src/api.py
```

The API will be available at `http://localhost:5000`.

### API Endpoints

#### Get all predictions for a week
```bash
curl "http://localhost:5000/predict?season=2024&week=5"
```

#### Get prediction for specific matchup
```bash
curl "http://localhost:5000/predict/matchup?home=KC&away=BUF&week=5"
```

#### Get model metrics
```bash
curl "http://localhost:5000/metrics"
```

#### Get list of teams
```bash
curl "http://localhost:5000/teams"
```

## Project Structure

```
nfl-prediction-model/
├── data/
│   ├── raw/              # Raw data from sources
│   ├── processed/        # Cleaned data
│   └── features/         # Engineered features
├── models/               # Saved models and metrics
├── predictions/          # Prediction outputs
├── src/                  # Source code
│   ├── fetch_data.py     # Data fetching
│   ├── build_features.py # Feature engineering
│   ├── train.py          # Model training
│   ├── predict.py        # Generate predictions
│   ├── dashboard.py      # Streamlit dashboard
│   ├── api.py            # Flask API
│   └── utils.py          # Helper functions
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
└── run_pipeline.py       # Main pipeline runner
```

## Typical Workflow

### Weekly Updates (During NFL Season)

1. **Update data** (run once per week before games):
   ```bash
   python run_pipeline.py --fetch --features
   ```

2. **Retrain model** (optional, if you want to include latest results):
   ```bash
   python run_pipeline.py --train
   ```

3. **Generate predictions** for upcoming week:
   ```bash
   python run_pipeline.py --predict --week <current_week>
   ```

4. **View in dashboard**:
   ```bash
   streamlit run src/dashboard.py
   ```

### Model Evaluation

Check model performance:

```bash
# View metrics file
cat models/metrics.json

# View feature importance
cat models/feature_importance.csv

# Compare models
cat models/model_comparison.csv
```

## Troubleshooting

### Data Fetching Issues

If data fetching fails:
- Check your internet connection
- Verify that nflfastR data is available for the season
- Try fetching a smaller date range in `config.yaml`

### Memory Issues

If you run out of memory:
- Reduce the date range in `config.yaml`
- Use a smaller `rolling_window` value
- Skip PBP data (comment out in `fetch_data.py`)

### Import Errors

If you get import errors:
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.11+)

## Configuration

Edit `config.yaml` to customize:

- **Data range**: `start_season` and `end_season`
- **Feature settings**: `rolling_window`, `min_games`
- **Model parameters**: Hyperparameters for each model
- **API settings**: Port, host, authentication

## Next Steps

- 📖 Read the full [README.md](README.md)
- 🔧 Customize model parameters in `config.yaml`
- 📊 Explore feature importance in `models/feature_importance.csv`
- 🚀 Deploy the API to production (AWS, Render, etc.)
- 📈 Track model performance over the season

## Getting Help

If you encounter issues:

1. Check the logs in console output
2. Review the configuration in `config.yaml`
3. Ensure all dependencies are installed
4. Verify data files exist in `data/raw/`

## Performance Targets

The model aims for:
- ✅ Accuracy: ≥ 65%
- ✅ ROC-AUC: ≥ 0.70
- ✅ Calibrated probabilities (low log loss)

Check your model's performance in the dashboard or via the API `/metrics` endpoint.

---

**Happy predicting! 🏈**

