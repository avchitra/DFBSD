import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import OutbreakPredictionModel
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_model_and_data():
    model = OutbreakPredictionModel('outbreaks.csv')
    evaluation = model.comprehensive_evaluation()
    return model, evaluation

def plot_confusion_matrix(evaluation):
    """
    Plot a confusion matrix given a comprehensive evaluation dictionary.

    Parameters
    ----------
    evaluation : dict
        A dictionary containing the model's comprehensive evaluation metrics,
        including the confusion matrix.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A Plotly figure representing the confusion matrix.
    """
    cm = evaluation['Confusion_Matrix']
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Low Risk', 'High Risk'],
                    y=['Low Risk', 'High Risk'],
                    text=cm,
                    color_continuous_scale='RdBu',
                    aspect='equal')
    fig.update_layout(title='Confusion Matrix')
    return fig

def plot_feature_importance(model):
    
    X, y = model.feature_engineering()
    clf, _, _ = model.create_model_pipeline()
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    fig = px.bar(importance, x='importance', y='feature', 
                 orientation='h', title='Top 15 Feature Importance')
    return fig

def plot_monthly_trends(data):
    
    monthly_counts = data.groupby('Month')['Outbreak_Risk'].value_counts().unstack()
    fig = px.line(monthly_counts, title='Monthly Outbreak Trends')
    return fig


def plot_geographical_distribution(data):
    """
    Plot the geographical distribution of high-risk outbreaks across the United States,
    given a Pandas DataFrame containing the outbreak data.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the outbreak data, with columns
        'State' and 'Outbreak_Risk'.

    Returns
    fig : plotly.graph_objs.Figure
        A Plotly figure representing the geographical distribution
        of high-risk outbreaks.
    """
    
    state_counts = data.groupby('State')['Outbreak_Risk'].value_counts().unstack()
    fig = px.choropleth(locations=state_counts.index,
                        locationmode="USA-states",
                        color=state_counts['High'],
                        scope="usa",
                        title='Geographical Distribution of High-Risk Outbreaks')
    return fig

def main():
    st.set_page_config(page_title="Outbreak Prediction Dashboard", layout="wide")

    col1, col2, col3 = st.columns(3)
    
    st.markdown("<h1 style='text-align: center; color: red;'>Outbreak Prediction Dashboard</h1>", unsafe_allow_html=True)    
    try:
        model, evaluation = load_model_and_data()
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Overview", "Model Performance", "Feature Analysis", "Geographical Insights"])
        
        if page == "Overview":
            st.header("Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Accuracy", 
                         f"{evaluation['ROC_AUC_Score']*100:.1f}%",
                         "ROC-AUC Score")
                
            with col2:
                st.metric("Average Precision",
                         f"{evaluation['Average_Precision']*100:.1f}%",
                         "For High-Risk Cases")
            
            st.subheader("Monthly Outbreak Trends")
            st.plotly_chart(plot_monthly_trends(model.data), use_container_width=True)
            
            st.subheader("Geographical Distribution")
            st.plotly_chart(plot_geographical_distribution(model.data), use_container_width=True)
            
        elif page == "Model Performance":
            st.header("Model Performance Metrics")
            
            # Classification Report
            st.subheader("Classification Report")
            st.text(evaluation['Classification_Report'])
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            st.plotly_chart(plot_confusion_matrix(evaluation), use_container_width=True)
            
        elif page == "Feature Analysis":
            st.header("Feature Importance Analysis")
            st.plotly_chart(plot_feature_importance(model), use_container_width=True)
            
            # Feature correlations
            st.subheader("Feature Correlations")
            X, _ = model.feature_engineering()
            corr = X.corr()
            fig = px.imshow(corr, title='Feature Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)
            
        elif page == "Geographical Insights":
            st.header("Geographical Analysis")
            
            # State-wise statistics
            state_stats = model.data.groupby('State').agg({
                'Illnesses': 'mean',
                'Hospitalizations': 'mean',
                'Outbreak_Risk': lambda x: (x == 'High').mean()
            }).round(2)
            
            st.subheader("State-wise Statistics")
            st.dataframe(state_stats)
            
            # Map visualization
            st.plotly_chart(plot_geographical_distribution(model.data), use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the 'outbreaks.csv' file is in the correct location and contains the required data.")

if __name__ == "__main__":
    main()