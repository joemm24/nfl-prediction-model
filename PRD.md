# 🏈 NFL Game Prediction Model — Product Requirements Document (PRD)

## 1. Overview

The **NFL Prediction Model** is a machine-learning–based system that forecasts the outcome of NFL games using historical data, team and player performance, coaching history, and contextual factors such as home-field advantage and weather conditions.
The goal is to provide accurate **win probabilities** for each matchup, available through both a **web dashboard** and a **REST API**.

---

## 2. Objectives

* Predict NFL game winners more accurately than naive baselines or Vegas odds.
* Combine multiple dimensions of data: offense, defense, coaching, home/away splits, and advanced metrics (ELO, EPA, DVOA).
* Automate weekly retraining and predictions.
* Present results in an interactive dashboard and API.

---

## 3. Key Features

### 3.1 Data Ingestion (ETL)

* Fetch data automatically from:

  * [nflfastR / NFLVerse](https://github.com/nflverse)
  * [Pro Football Reference](https://www.pro-football-reference.com/)
  * Optional: Football Outsiders (DVOA), FiveThirtyEight (ELO)
* Clean and store in structured format (CSV/Parquet or database).
* Scheduled weekly updates before each game week.

### 3.2 Feature Engineering

* Compute aggregated and differential stats:

  * Offensive/defensive yards, efficiency, turnovers, sacks.
  * Rolling averages (last 3–5 games).
  * Home/away performance splits.
  * Coaching and team head-to-head metrics.
  * Advanced stats (ELO, EPA, DVOA).
  * Environmental factors (weather, surface, dome indicator).
* Output: unified "features" dataset ready for modeling.

### 3.3 Modeling

* Algorithms: Logistic Regression → Random Forest → XGBoost (select best).
* Target: `home_team_win` (binary classification).
* Metrics:

  * Accuracy
  * Log Loss
  * ROC-AUC
  * Calibration curve
* Cross-validation across seasons to prevent overfitting.
* Retraining weekly with new data.

### 3.4 Predictions

* Input: upcoming weekly schedule.
* Output:

  * Win probability for each team.
  * Confidence score.
* Export formats:

  * JSON (API)
  * CSV (analyst use)
  * Live dashboard display.

### 3.5 Dashboard

* Built with **Streamlit**.
* Features:

  * Filter by team, week, or season.
  * Show matchup cards with probabilities.
  * Confidence visualizations (bar or gauge charts).
  * Track model performance over time.

### 3.6 API Service

* Built with **Flask**.
* Endpoints:

  * `GET /predict` — returns probabilities for requested matchups.
  * `GET /metrics` — returns model accuracy, log loss, and metadata.
* Optional authentication via API key.
* Deployable on AWS Lambda, EC2, or Render.

---

## 4. Tech Stack

| Layer           | Technology                      |
| --------------- | ------------------------------- |
| Language        | Python 3.11+                    |
| Data            | pandas, requests, pyarrow       |
| Modeling        | scikit-learn, XGBoost, LightGBM |
| Storage         | CSV / SQLite / PostgreSQL       |
| API             | Flask                           |
| Dashboard       | Streamlit                       |
| Orchestration   | npm scripts + Python virtualenv |
| Version Control | GitHub                          |
| CI/CD           | GitHub Actions                  |

---

## 5. Data Model

| Entity            | Description                                 |
| ----------------- | ------------------------------------------- |
| `games`           | Historical schedules and results            |
| `teams`           | Team metadata (name, division, coach)       |
| `stats_team_game` | Offensive & defensive metrics per game      |
| `features`        | Engineered dataset for ML training          |
| `predictions`     | Model outputs (win probability, confidence) |

---

## 6. System Architecture

```
            ┌────────────────────┐
            │  Data Sources      │
            │ (NFLfastR, PFR)    │
            └────────┬───────────┘
                     │
              [fetch_data.py]
                     │
            ┌────────▼────────┐
            │  ETL + Features │
            │ (build_features.py)
            └────────┬────────┘
                     │
              [train.py / model.pkl]
                     │
        ┌────────────┴────────────┐
        │                         │
 [Streamlit Dashboard]     [Flask API]
        │                         │
        └────────────┬────────────┘
                     │
             [Predictions Output]
```

---

## 7. Success Metrics

* ✅ Model accuracy ≥ **65%** on test set
* ✅ ROC-AUC ≥ **0.70**
* ✅ Automated weekly retraining
* ✅ Dashboard loads in under 3 seconds
* ✅ Reliable JSON/CSV export for all weekly matchups

---

## 8. Milestones & Timeline

| Phase     | Deliverable                               | Duration |
| --------- | ----------------------------------------- | -------- |
| Phase 1   | Data pipeline (ETL + feature builder)     | 2 weeks  |
| Phase 2   | Baseline model (Logistic Regression)      | 1 week   |
| Phase 3   | Advanced model tuning (XGBoost, LightGBM) | 1 week   |
| Phase 4   | Streamlit dashboard (MVP UI)              | 1 week   |
| Phase 5   | Flask API + deployment                    | 1 week   |
| Phase 6   | Testing, CI/CD, documentation             | 1 week   |
| **Total** | **MVP in ~6–7 weeks**                     |          |

---

## 9. Future Enhancements

* Add player-level stats and injury data.
* Incorporate Vegas lines for EV calculations.
* Build ensemble model combining ELO + ML.
* Add live in-game win probability updates.
* Include user authentication and prediction history.
* Publish weekly accuracy summaries to dashboard or Slack.

