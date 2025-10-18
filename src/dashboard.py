"""
Streamlit Dashboard - NFL Prediction Model
Interactive web dashboard for visualizing NFL game predictions
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, load_metrics
from src.predict import NFLPredictor


# Page configuration
st.set_page_config(
    page_title="NFL Game Predictions",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# NFL Team Colors and Logos
NFL_TEAMS = {
    'ARI': {'name': 'Cardinals', 'color': '#97233F', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/ari.png'},
    'ATL': {'name': 'Falcons', 'color': '#A71930', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/atl.png'},
    'BAL': {'name': 'Ravens', 'color': '#241773', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/bal.png'},
    'BUF': {'name': 'Bills', 'color': '#00338D', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/buf.png'},
    'CAR': {'name': 'Panthers', 'color': '#0085CA', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/car.png'},
    'CHI': {'name': 'Bears', 'color': '#0B162A', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/chi.png'},
    'CIN': {'name': 'Bengals', 'color': '#FB4F14', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/cin.png'},
    'CLE': {'name': 'Browns', 'color': '#311D00', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/cle.png'},
    'DAL': {'name': 'Cowboys', 'color': '#003594', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/dal.png'},
    'DEN': {'name': 'Broncos', 'color': '#FB4F14', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/den.png'},
    'DET': {'name': 'Lions', 'color': '#0076B6', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/det.png'},
    'GB': {'name': 'Packers', 'color': '#203731', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/gb.png'},
    'HOU': {'name': 'Texans', 'color': '#03202F', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/hou.png'},
    'IND': {'name': 'Colts', 'color': '#002C5F', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/ind.png'},
    'JAX': {'name': 'Jaguars', 'color': '#006778', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/jax.png'},
    'KC': {'name': 'Chiefs', 'color': '#E31837', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/kc.png'},
    'LA': {'name': 'Rams', 'color': '#003594', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/lar.png'},
    'LAC': {'name': 'Chargers', 'color': '#0080C6', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/lac.png'},
    'LV': {'name': 'Raiders', 'color': '#000000', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/lv.png'},
    'MIA': {'name': 'Dolphins', 'color': '#008E97', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/mia.png'},
    'MIN': {'name': 'Vikings', 'color': '#4F2683', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/min.png'},
    'NE': {'name': 'Patriots', 'color': '#002244', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/ne.png'},
    'NO': {'name': 'Saints', 'color': '#D3BC8D', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/no.png'},
    'NYG': {'name': 'Giants', 'color': '#0B2265', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png'},
    'NYJ': {'name': 'Jets', 'color': '#125740', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png'},
    'PHI': {'name': 'Eagles', 'color': '#004C54', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/phi.png'},
    'PIT': {'name': 'Steelers', 'color': '#FFB612', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/pit.png'},
    'SEA': {'name': 'Seahawks', 'color': '#002244', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/sea.png'},
    'SF': {'name': '49ers', 'color': '#AA0000', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/sf.png'},
    'TB': {'name': 'Buccaneers', 'color': '#D50A0A', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/tb.png'},
    'TEN': {'name': 'Titans', 'color': '#0C2340', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/ten.png'},
    'WAS': {'name': 'Commanders', 'color': '#5A1414', 'logo': 'https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png'},
}


# Custom CSS for compact, beautiful cards
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .game-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .team-section {
        text-align: center;
        padding: 0.5rem;
    }
    .team-logo {
        width: 60px;
        height: 60px;
        margin: 0 auto;
    }
    .team-abbr {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin: 0.5rem 0;
    }
    .win-prob {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.25rem 0;
    }
    .confidence-badge {
        background-color: rgba(255, 255, 255, 0.2);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        color: white;
        display: inline-block;
        margin-top: 0.25rem;
    }
    .vs-text {
        font-size: 2rem;
        color: white;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }
    .game-date {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


class NFLDashboard:
    """NFL Prediction Dashboard"""
    
    def __init__(self):
        """Initialize dashboard"""
        self.config = load_config()
        self.predictor = NFLPredictor()
        self.predictions_dir = self.config['predictions']['output_dir']
        self.model_dir = self.config['model']['save_dir']
    
    def load_latest_predictions(self) -> pd.DataFrame:
        """Load the most recent predictions"""
        predictions_path = os.path.join(self.predictions_dir, "predictions_latest.csv")
        
        if not os.path.exists(predictions_path):
            return pd.DataFrame()
        
        return pd.read_csv(predictions_path)
    
    def load_model_metrics(self) -> dict:
        """Load model performance metrics"""
        metrics_path = os.path.join(self.model_dir, "metrics.json")
        
        if not os.path.exists(metrics_path):
            return {}
        
        return load_metrics(metrics_path)
    
    def create_compact_matchup_card(self, game: pd.Series, col):
        """Create a compact, beautiful matchup card"""
        home_team = game['home_team']
        away_team = game['away_team']
        home_prob = game['home_win_prob']
        away_prob = game['away_win_prob']
        confidence = game['confidence']
        predicted_winner = game['predicted_winner']
        
        # Get team info
        home_info = NFL_TEAMS.get(home_team, {'name': home_team, 'color': '#cccccc', 'logo': ''})
        away_info = NFL_TEAMS.get(away_team, {'name': away_team, 'color': '#cccccc', 'logo': ''})
        
        with col:
            # Create probability bar with team colors
            home_color = home_info['color']
            away_color = away_info['color']
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=['Probability'],
                x=[home_prob * 100],
                name=home_team,
                orientation='h',
                marker=dict(color=home_color),
                text=[f'{home_prob:.1%}'],
                textposition='inside',
                textfont=dict(size=14, color='white'),
                hovertemplate=f'{home_team}: {home_prob:.1%}<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                y=['Probability'],
                x=[away_prob * 100],
                name=away_team,
                orientation='h',
                marker=dict(color=away_color),
                text=[f'{away_prob:.1%}'],
                textposition='inside',
                textfont=dict(size=14, color='white'),
                hovertemplate=f'{away_team}: {away_prob:.1%}<extra></extra>'
            ))
            
            fig.update_layout(
                barmode='stack',
                height=80,
                showlegend=False,
                xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            
            # Display matchup
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown(f"""
                    <div class="team-section">
                        <img src="{away_info['logo']}" class="team-logo" />
                        <div class="team-abbr">{away_team}</div>
                        <div class="win-prob" style="color: {'#00ff00' if away_prob > 0.5 else '#ff6b6b'};">{away_prob:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="vs-text">@</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="team-section">
                        <img src="{home_info['logo']}" class="team-logo" />
                        <div class="team-abbr">{home_team}</div>
                        <div class="win-prob" style="color: {'#00ff00' if home_prob > 0.5 else '#ff6b6b'};">{home_prob:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Probability bar
            st.plotly_chart(fig, use_container_width=True)
            
            # Info row
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"<div style='text-align:center; color:white;'><strong>Winner</strong><br/>{predicted_winner}</div>", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<div style='text-align:center; color:white;'><strong>Confidence</strong><br/>{confidence:.1%}</div>", unsafe_allow_html=True)
            with col_c:
                game_date = game.get('gameday', 'TBD')
                if pd.notna(game_date):
                    st.markdown(f"<div style='text-align:center; color:white;'><strong>Date</strong><br/>{game_date}</div>", unsafe_allow_html=True)
            
            st.markdown("---")
    
    def run(self):
        """Run the dashboard"""
        # Header
        st.markdown("<h1 class='main-header'>🏈 NFL Game Predictions</h1>", 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Model info
            st.subheader("Model Information")
            metrics = self.load_model_metrics()
            
            if metrics:
                st.metric("Model Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
                st.metric("Log Loss", f"{metrics.get('log_loss', 0):.3f}")
                
                if 'last_updated' in metrics:
                    st.caption(f"Last updated: {metrics['last_updated'][:10]}")
            else:
                st.info("No model metrics available")
            
            st.divider()
            
            # Prediction controls
            st.subheader("Generate Predictions")
            
            current_year = datetime.now().year
            season = st.number_input("Season", min_value=2010, max_value=current_year + 1,
                                    value=current_year, step=1)
            week = st.number_input("Week", min_value=1, max_value=18, value=6, step=1)
            
            if st.button("🔮 Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    try:
                        predictions = self.predictor.predict(season, week)
                        st.success(f"Generated {len(predictions)} predictions!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            st.divider()
            
            # Filters
            st.subheader("Filters")
            show_high_confidence = st.checkbox("Show only high confidence (>30%)", value=False)
            sort_by = st.selectbox("Sort by", 
                                   ["Confidence (High to Low)", 
                                    "Confidence (Low to High)",
                                    "Home Team Win Prob"])
        
        # Main content
        predictions = self.load_latest_predictions()
        
        if predictions.empty:
            st.info("👈 No predictions available. Use the sidebar to generate predictions.")
            
            # Show sample data structure
            with st.expander("ℹ️ How to use this dashboard"):
                st.markdown("""
                ### Getting Started
                
                1. **Generate Predictions**: Use the sidebar to select a season and week, 
                   then click "Generate Predictions"
                2. **View Results**: Predictions will appear as cards showing win probabilities
                3. **Filter Results**: Use the filters to focus on specific games
                4. **Model Performance**: Check the sidebar for model accuracy metrics
                
                ### Understanding the Predictions
                
                - **Win Probability**: Chance of each team winning (adds up to 100%)
                - **Confidence**: How certain the model is (higher = more certain)
                - **Predicted Winner**: Team with >50% win probability
                """)
            
            return
        
        # Display summary
        st.subheader(f"📊 Week {predictions['week'].iloc[0]} Predictions "
                    f"({predictions['season'].iloc[0]} Season)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Games", len(predictions))
        
        with col2:
            avg_confidence = predictions['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            high_conf = (predictions['confidence'] > 0.3).sum()
            st.metric("High Confidence Games", high_conf)
        
        with col4:
            close_games = (predictions['confidence'] < 0.1).sum()
            st.metric("Toss-up Games", close_games)
        
        st.divider()
        
        # Apply filters
        filtered_predictions = predictions.copy()
        
        if show_high_confidence:
            filtered_predictions = filtered_predictions[filtered_predictions['confidence'] > 0.3]
        
        # Sort
        if sort_by == "Confidence (High to Low)":
            filtered_predictions = filtered_predictions.sort_values('confidence', ascending=False)
        elif sort_by == "Confidence (Low to High)":
            filtered_predictions = filtered_predictions.sort_values('confidence', ascending=True)
        elif sort_by == "Home Team Win Prob":
            filtered_predictions = filtered_predictions.sort_values('home_win_prob', ascending=False)
        
        # Display predictions in 2-column layout
        st.subheader(f"🎯 Game Predictions ({len(filtered_predictions)} games)")
        
        if filtered_predictions.empty:
            st.warning("No games match the current filters.")
        else:
            # Display predictions 2 per row
            for i in range(0, len(filtered_predictions), 2):
                col1, col2 = st.columns(2)
                
                # First prediction in row
                game1 = filtered_predictions.iloc[i]
                self.create_compact_matchup_card(game1, col1)
                
                # Second prediction in row (if exists)
                if i + 1 < len(filtered_predictions):
                    game2 = filtered_predictions.iloc[i + 1]
                    self.create_compact_matchup_card(game2, col2)
        
        # Additional visualizations
        st.divider()
        st.subheader("📈 Prediction Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution
            fig = px.histogram(predictions, x='confidence', nbins=20,
                             title='Confidence Distribution',
                             labels={'confidence': 'Confidence', 'count': 'Number of Games'})
            fig.update_traces(marker_color='#1f77b4')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Win probability distribution
            fig = px.histogram(predictions, x='home_win_prob', nbins=20,
                             title='Home Team Win Probability Distribution',
                             labels={'home_win_prob': 'Win Probability', 'count': 'Number of Games'})
            fig.update_traces(marker_color='#2ca02c')
            st.plotly_chart(fig, use_container_width=True)
        
        # Download predictions
        st.divider()
        st.subheader("💾 Download Predictions")
        
        csv = predictions.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"nfl_predictions_week_{predictions['week'].iloc[0]}.csv",
            mime="text/csv"
        )


def main():
    """Main function to run dashboard"""
    dashboard = NFLDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
