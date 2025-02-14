import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

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
        
    def plot_data_distribution(self, cdc_data):
        """
        Plot distribution of data sources
        
        Parameters:
        -----------
        cdc_data : pandas.DataFrame
            DataFrame containing CDC outbreak data
        """
        # First, let's examine what columns we have
        print("CDC data columns:", cdc_data.columns.tolist())
        
        fig, ax1 = plt.subplots(figsize=(15, 6))
        
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
        
        plt.tight_layout()
        plt.savefig(self.save_path / 'data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    def plot_model_performance(self, true_positives, false_negatives, false_positives, true_negatives, feature_importance):
        """
        Plot model performance using a confusion matrix, ROC curve, and feature importance.
        """
        # Create figure and axes - fix the unpacking
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Confusion Matrix
        cm = np.array([[true_negatives, false_positives], 
                    [false_negatives, true_positives]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.set_xticklabels(['Negative', 'Positive'])
        ax1.set_yticklabels(['Negative', 'Positive'])
        
        # ROC Curve
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        ax2.plot([0, fpr, 1], [0, tpr, 1], color=self.colors[1], marker='o', linestyle='-')
        ax2.plot([0, 1], [0, 1], 'k--')  # Random classifier reference line
        ax2.set_title('ROC Curve')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')

        # Feature Importance
        sorted_feat_imp = feature_importance.sort_values(by='importance', ascending=True)
        sorted_feat_imp.plot(kind='barh', x='feature', y='importance', ax=ax3, color=self.colors[3])
        ax3.set_title('Feature Importance')

        # Apply styling to each axis
        for ax in [ax1, ax2, ax3]:
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
        
    def generate_all_figures(self, cdc_data, model_results, detection_results):
        """Generate all figures for the project"""
        print("Generating data distribution plots...")
        self.plot_data_distribution(cdc_data)
        
        print("Generating model performance plots...")
        self.plot_model_performance(
            model_results['true_positives'],
            model_results['false_negatives'],
            model_results['false_positives'],
            model_results['true_negatives'],
            model_results['feature_importance']
        )
        
        print("Generating detection timeline comparison...")
        self.create_timeline_comparison(detection_results)
        
        print(f"All figures saved to {self.save_path}")

# Example usage
if __name__ == "__main__":
#   try:
        visualizer = ProjectVisualizer()
        
        # Load your data here
        cdc_data = pd.read_csv('outbreaks.csv')
        
        feature_importance = pd.DataFrame({
        "feature": [
            "Hospitalizations", "Location_encoded", "Status_encoded", 
            "State_encoded", "Food_encoded", "Fatalities", "Years_Ago", 
            "Year", "Month_cos", "Month_sin"
        ],
        "importance": [0.414242, 0.239853, 0.209918, 0.071430, 0.022342, 
                       0.017207, 0.010862, 0.008702, 0.003602, 0.001844]
        })



        # Create sample model results dictionary
        model_results = {
            'true_negatives': 1580,
            'false_positives': 413,
            'false_negatives': 647,
            'true_positives': 1184,
            'feature_importance': feature_importance,
        }
        
        detection_results = pd.DataFrame({
            'outbreak_date': pd.date_range(start='2024-01-01', periods=10),
            'traditional_detection_days': np.random.randint(5, 15, 10),
            'ml_detection_days': np.random.randint(2, 8, 10)
        })
        
        visualizer.generate_all_figures(cdc_data, model_results, detection_results)
        
#    except Exception as e:
#        print(f"An error occurred: {str(e)}")
#        # Print more detailed information about the data
#        print("\nData info:")
#        if 'cdc_data' in locals():
#            print("\nCDC data info:")
#            print(cdc_data.info())