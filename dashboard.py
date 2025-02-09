# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from model import OutbreakPredictor

class OutbreakDashboard:
    def __init__(self, model_path):
        self.model = OutbreakPredictor(model_path)
        self.load_data()
    
    def load_data(self):
        """Load and prepare data for visualization"""
        self.outbreak_data = pd.read_csv('outbreaks.csv')
        self.predictions = self.make_predictions()
    
    def make_predictions(self):
        """Generate predictions for visualization"""
        X = self.model.preprocess_features(self.outbreak_data)
        probas = self.model.predict_proba(X)
        
        predictions_df = self.outbreak_data.copy()
        predictions_df['Risk_Score'] = probas
        predictions_df['Risk_Level'] = pd.qcut(
            probas, 
            q=4, 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        return predictions_df
    
    def run_dashboard(self):
        """Create and run the Streamlit dashboard"""
        st.title('Foodborne Illness Outbreak Risk Dashboard')
        
        # Sidebar filters
        st.sidebar.header('Filters')
        selected_state = st.sidebar.multiselect(
            'Select States',
            options=self.outbreak_data['State'].unique()
        )
        
        # Filter data based on selection
        if selected_state:
            filtered_data = self.predictions[
                self.predictions['State'].isin(selected_state)
            ]
        else:
            filtered_data = self.predictions
        
        # Risk distribution
        st.header('Risk Distribution')
        fig_risk = px.histogram(
            filtered_data, 
            x='Risk_Score',
            nbins=50,
            title='Distribution of Risk Scores'
        )
        st.plotly_chart(fig_risk)
        
        # Geographical distribution
        st.header('Geographical Risk Distribution')
        fig_geo = px.choropleth(
            filtered_data.groupby('State')['Risk_Score'].mean().reset_index(),
            locations='State',
            locationmode='USA-states',
            color='Risk_Score',
            scope='usa',
            title='Average Risk Score by State'
        )
        st.plotly_chart(fig_geo)
        
        # Feature importance
        st.header('Model Feature Importance')
        fig_importance = px.bar(
            x=self.model.feature_importance.values,
            y=self.model.feature_importance.index,
            orientation='h',
            title='Feature Importance'
        )
        st.plotly_chart(fig_importance)
        
        # High risk cases
        st.header('High Risk Cases')
        high_risk = filtered_data[
            filtered_data['Risk_Level'].isin(['High', 'Very High'])
        ].sort_values('Risk_Score', ascending=False)
        st.dataframe(high_risk)

def main():
    dashboard = OutbreakDashboard('outbreak_model.pkl')
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()