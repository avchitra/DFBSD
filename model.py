import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    average_precision_score
)
from imblearn.over_sampling import SMOTE
import warnings

class OutbreakPredictionModel:
    def __init__(self, data_path):
        warnings.filterwarnings('ignore')
        self.data = pd.read_csv(data_path)
        self.preprocess_data()
    
    def preprocess_data(self):
        # Handle missing values
        self.data = self.data.fillna('Unknown')
        
        # Ensure numeric columns are properly converted
        numeric_columns = ['Illnesses', 'Hospitalizations', 'Fatalities']
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)
        
        # Convert Month to numeric if it's string
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 
            'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9,
            'October': 10, 'November': 11, 'December': 12
        }
        if isinstance(self.data['Month'].iloc[0], str):
            self.data['Month'] = self.data['Month'].map(month_map)
        
        # Convert Year to numeric
        self.data['Year'] = pd.to_numeric(self.data['Year'], errors='coerce').fillna(0)
        
        # Create outbreak risk category based on illnesses and hospitalizations
        illness_threshold = self.data['Illnesses'].quantile(0.75)
        hosp_threshold = self.data['Hospitalizations'].mean()
        
        self.data['Outbreak_Risk'] = np.where(
            (self.data['Illnesses'] > illness_threshold) | 
            (self.data['Hospitalizations'] > hosp_threshold) |
            (self.data['Fatalities'] > 0),
            'High', 'Low'
        )
    
    def feature_engineering(self):
        # Select features present in the data
        predictive_features = [
            col for col in ['State', 'Location', 'Food', 'Ingredient', 
                           'Species', 'Serotype/Genotype', 'Month', 'Year'] 
            if col in self.data.columns
        ]
        
        # Create dummy variables for categorical features
        X = pd.get_dummies(self.data[predictive_features])
        
        # Add normalized numerical features
        X['Illnesses_Normalized'] = (self.data['Illnesses'] - self.data['Illnesses'].mean()) / self.data['Illnesses'].std()
        X['Hospitalizations_Normalized'] = (self.data['Hospitalizations'] - self.data['Hospitalizations'].mean()) / (self.data['Hospitalizations'].std() + 1e-6)
        X['Fatalities_Normalized'] = (self.data['Fatalities'] - self.data['Fatalities'].mean()) / (self.data['Fatalities'].std() + 1e-6)
        
        # Create target variable
        y = (self.data['Outbreak_Risk'] == 'High').astype(int)
        
        return X, y
    
    def create_model_pipeline(self):
        X, y = self.feature_engineering()
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE for handling class imbalance
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Create and train the model
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        clf.fit(X_train_resampled, y_train_resampled)
        
        return clf, X_test, y_test
    
    def evaluate_model(self):
        """
        Evaluate the model's performance using comprehensive metrics.

        The metrics used include the Confusion Matrix, Classification Report,
        ROC AUC Score, and Average Precision. The model is trained using the
        create_model_pipeline method and evaluated on the test set.

        Returns
        -------
        dict
            A dictionary containing the comprehensive metrics. The keys are
            'Confusion_Matrix', 'Classification_Report', 'ROC_AUC_Score',
            and 'Average_Precision'.
        """
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
    
    def get_data(self):
        """Return the processed dataframe"""
        return self.data