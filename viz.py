import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

class ProjectVisualizer:
    def __init__(self, save_path="project_figures/"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Set consistent style for all plots
        plt.style.use('default')
        self.colors = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71']
        
        # Set global style parameters
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        
    def set_style(self, ax):
        """Apply consistent styling to matplotlib plots"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
    def plot_data_distribution(self, cdc_data, twitter_data):
        """
        Plot distribution of data sources
        
        Parameters:
        -----------
        cdc_data : pandas.DataFrame
            DataFrame containing CDC outbreak data
        twitter_data : pandas.DataFrame
            DataFrame containing Twitter data
        """
        # First, let's examine what columns we have
        print("CDC data columns:", cdc_data.columns.tolist())
        print("Twitter data columns:", twitter_data.columns.tolist())
        
        # Create subplots based on available data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # CDC Outbreaks by State (if available)
        if 'State' in cdc_data.columns:
            state_counts = cdc_data['State'].value_counts().head(10)
            state_counts.plot(kind='bar', ax=ax1, color=self.colors[0])
            ax1.set_title('Top 10 States by Number of Outbreaks\n(CDC Data)')
            ax1.set_xlabel('State')
            ax1.set_ylabel('Number of Outbreaks')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            # Plot overall distribution of a numerical column if available
            num_cols = cdc_data.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                sns.histplot(data=cdc_data, x=num_cols[0], ax=ax1, color=self.colors[0])
                ax1.set_title(f'Distribution of {num_cols[0]}\n(CDC Data)')
            else:
                ax1.text(0.5, 0.5, 'No suitable columns found for visualization', 
                        ha='center', va='center')
        
        self.set_style(ax1)
        
        # Twitter Data Distribution
        # Try to find a suitable numerical column for visualization
        twitter_num_cols = twitter_data.select_dtypes(include=[np.number]).columns
        if len(twitter_num_cols) > 0:
            sns.histplot(data=twitter_data, x=twitter_num_cols[0], ax=ax2, color=self.colors[2])
            ax2.set_title(f'Distribution of {twitter_num_cols[0]}\n(Twitter Data)')
        else:
            # If no numerical columns, try categorical
            cat_cols = twitter_data.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                twitter_data[cat_cols[0]].value_counts().head(10).plot(
                    kind='bar', ax=ax2, color=self.colors[2])
                ax2.set_title(f'Top 10 {cat_cols[0]} Values\n(Twitter Data)')
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax2.text(0.5, 0.5, 'No suitable columns found for visualization', 
                        ha='center', va='center')
        
        self.set_style(ax2)
        
        plt.tight_layout()
        plt.savefig(self.save_path / 'data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_model_performance(self, model_results):
        """Plot model performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(model_results['true'], model_results['pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        
        # ROC Curve
        fpr, tpr = model_results['roc_curve']
        axes[0,1].plot(fpr, tpr, color=self.colors[1])
        axes[0,1].plot([0, 1], [0, 1], 'k--')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        
        # Feature Importance
        feat_imp = model_results['feature_importance'].sort_values(ascending=True)
        feat_imp.plot(kind='barh', ax=axes[1,0], color=self.colors[3])
        axes[1,0].set_title('Feature Importance')
        
        # Performance Over Time
        time_perf = model_results['time_performance']
        axes[1,1].plot(time_perf.index, time_perf['f1_score'], color=self.colors[2])
        axes[1,1].set_title('Model Performance Over Time')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('F1 Score')
        
        for ax in axes.flat:
            self.set_style(ax)
            
        plt.tight_layout()
        plt.savefig(self.save_path / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_timeline_comparison(self, detection_results):
        """Create timeline comparing traditional vs. ML detection"""
        fig = go.Figure()
        
        # Add traditional detection timeline
        fig.add_trace(go.Scatter(
            x=detection_results['outbreak_date'],
            y=detection_results['traditional_detection_days'],
            name='Traditional Detection',
            mode='markers+lines',
            line=dict(color=self.colors[0])
        ))
        
        # Add ML detection timeline
        fig.add_trace(go.Scatter(
            x=detection_results['outbreak_date'],
            y=detection_results['ml_detection_days'],
            name='ML Detection',
            mode='markers+lines',
            line=dict(color=self.colors[1])
        ))
        
        fig.update_layout(
            title='Detection Time Comparison',
            xaxis_title='Outbreak Date',
            yaxis_title='Days to Detection',
            template='plotly_white'
        )
        
        fig.write_image(str(self.save_path / 'detection_timeline.png'))
        
    def generate_all_figures(self, cdc_data, twitter_data, model_results, detection_results):
        """Generate all figures for the project"""
        print("Generating data distribution plots...")
        self.plot_data_distribution(cdc_data, twitter_data)
        
        print("Generating model performance plots...")
        self.plot_model_performance(model_results)
        
        print("Generating detection timeline comparison...")
        self.create_timeline_comparison(detection_results)
        
        print(f"All figures saved to {self.save_path}")

# Example usage
if __name__ == "__main__":
    try:
        visualizer = ProjectVisualizer()
        
        # Load your data here
        cdc_data = pd.read_csv('outbreaks.csv')
        twitter_data = pd.read_pickle('TWEET-FID/LREC_BSC/train.p')
        
        # Create sample model results dictionary
        model_results = {
            'true': np.array([0, 1, 0, 1, 1]),
            'pred': np.array([0, 1, 0, 0, 1]),
            'roc_curve': (np.array([0, 0.5, 1]), np.array([0, 0.7, 1])),
            'feature_importance': pd.Series({'feature1': 0.3, 'feature2': 0.7}),
            'time_performance': pd.DataFrame({
                'f1_score': [0.8, 0.85, 0.82]
            }, index=pd.date_range('2024-01-01', periods=3))
        }
        
        detection_results = pd.DataFrame({
            'outbreak_date': pd.date_range(start='2024-01-01', periods=10),
            'traditional_detection_days': np.random.randint(5, 15, 10),
            'ml_detection_days': np.random.randint(2, 8, 10)
        })
        
        visualizer.generate_all_figures(cdc_data, twitter_data, model_results, detection_results)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Print more detailed information about the data
        print("\nData info:")
        if 'cdc_data' in locals():
            print("\nCDC data info:")
            print(cdc_data.info())
        if 'twitter_data' in locals():
            print("\nTwitter data info:")
            print(twitter_data.info())