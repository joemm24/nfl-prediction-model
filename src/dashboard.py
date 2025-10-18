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


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .matchup-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .team-name {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prob-high {
        color: #00cc00;
        font-weight: bold;
    }
    .prob-low {
        color: #cc0000;
        font-weight: bold;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
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
    
    def create_gauge_chart(self, value: float, title: str) -> go.Figure:
        """Create a gauge chart for probability"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value * 100,
            title={'text': title, 'font': {'size': 20}},
            number={'suffix': "%", 'font': {'size': 30}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#1f77b4"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#ffcccc'},
                    {'range': [50, 70], 'color': '#ffffcc'},
                    {'range': [70, 100], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    
    def create_probability_bar(self, home_team: str, away_team: str,
                              home_prob: float, away_prob: float) -> go.Figure:
        """Create horizontal bar chart for win probabilities"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=[f'{away_team} @ {home_team}'],
            x=[home_prob * 100],
            name=home_team,
            orientation='h',
            marker=dict(color='#1f77b4'),
            text=[f'{home_prob:.1%}'],
            textposition='inside',
            textfont=dict(size=16, color='white')
        ))
        
        fig.add_trace(go.Bar(
            y=[f'{away_team} @ {home_team}'],
            x=[away_prob * 100],
            name=away_team,
            orientation='h',
            marker=dict(color='#ff7f0e'),
            text=[f'{away_prob:.1%}'],
            textposition='inside',
            textfont=dict(size=16, color='white')
        ))
        
        fig.update_layout(
            barmode='stack',
            height=150,
            showlegend=True,
            xaxis=dict(range=[0, 100], showticklabels=False),
            yaxis=dict(showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def display_matchup_card(self, game: pd.Series):
        """Display a single game prediction card"""
        home_team = game['home_team']
        away_team = game['away_team']
        home_prob = game['home_win_prob']
        away_prob = game['away_win_prob']
        confidence = game['confidence']
        predicted_winner = game['predicted_winner']
        
        with st.container():
            # Header
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown(f"<div class='team-name'>{away_team}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='{'prob-high' if away_prob > 0.5 else 'prob-low'}'>"
                          f"{away_prob:.1%}</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='text-align: center; font-size: 2rem;'>@</div>", 
                          unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"<div class='team-name'>{home_team}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='{'prob-high' if home_prob > 0.5 else 'prob-low'}'>"
                          f"{home_prob:.1%}</div>", unsafe_allow_html=True)
            
            # Probability bar
            fig = self.create_probability_bar(home_team, away_team, home_prob, away_prob)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Winner", predicted_winner)
            
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col3:
                if 'gameday' in game and pd.notna(game['gameday']):
                    st.metric("Game Day", game['gameday'])
            
            st.divider()
    
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
            week = st.number_input("Week", min_value=1, max_value=18, value=1, step=1)
            
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
        
        # Display predictions
        st.subheader(f"🎯 Game Predictions ({len(filtered_predictions)} games)")
        
        if filtered_predictions.empty:
            st.warning("No games match the current filters.")
        else:
            for idx, game in filtered_predictions.iterrows():
                self.display_matchup_card(game)
        
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

