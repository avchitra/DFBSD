import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score
)
from sklearn.feature_selection import mutual_info_classif
import warnings

class OutbreakPredictionModel:
    def __init__(self, data_path):
        warnings.filterwarnings('ignore')
        self.data = pd.read_csv(data_path)
        self.preprocess_data()
    
    def preprocess_data(self):
        # Explicit date parsing
        self.data['Date'] = pd.to_datetime(
        self.data['Year'].astype(str) + '-' + 
        self.data['Month'].astype(str) + '-01'
    )
        
        # Advanced season mapping
        month_map = {
        'January': 1, 'February': 2, 'March': 3, 
        'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9,
        'October': 10, 'November': 11, 'December': 12
        }
        
        if isinstance(self.data['Month'].iloc[0], str):
            self.data['Month'] = self.data['Month'].map(month_map)

        # More sophisticated outbreak risk categorization
        self.data['Outbreak_Risk'] = np.where(
            (self.data['Illnesses'] > self.data['Illnesses'].quantile(0.75)) & 
            (self.data['Hospitalizations'] > 0),
            'High', 'Low'
        )
    
    def feature_engineering(self):
        # Remove 'Season' if not in columns
        predictive_features = [
            col for col in ['State', 'Location', 'Food', 'Ingredient', 
                        'Season', 'Species', 'Serotype/Genotype'] 
            if col in self.data.columns
        ]
    
        X = pd.get_dummies(self.data[predictive_features])
        y = (self.data['Outbreak_Risk'] == 'High').astype(int)
    
    # Rest of the method remains the same
        
        # Feature importance
        feature_importance = mutual_info_classif(X, y)
        feature_importance_dict = dict(zip(X.columns, feature_importance))
        
        return X, y, feature_importance_dict
    
    def create_model_pipeline(self):
        X, y, feature_importance = self.feature_engineering()
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Advanced Random Forest with feature weighting
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        clf.fit(X_train, y_train)
        
        return clf, X_test, y_test
    
    def comprehensive_evaluation(self):
        clf, X_test, y_test = self.create_model_pipeline()
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Comprehensive Metrics
        return {
            'Confusion_Matrix': confusion_matrix(y_test, y_pred),
            'Classification_Report': classification_report(y_test, y_pred),
            'ROC_AUC_Score': roc_auc_score(y_test, y_pred_proba),
            'Average_Precision': average_precision_score(y_test, y_pred_proba)
        }

def main():
    model = OutbreakPredictionModel('outbreaks.csv')
    
    # Evaluate model
    evaluation = model.comprehensive_evaluation()
    
    print("Confusion Matrix:\n", evaluation['Confusion_Matrix'])
    print("\nClassification Report:\n", evaluation['Classification_Report'])
    print(f"\nROC AUC Score: {evaluation['ROC_AUC_Score']:.4f}")
    print(f"Average Precision: {evaluation['Average_Precision']:.4f}")

if __name__ == "__main__":
    main()