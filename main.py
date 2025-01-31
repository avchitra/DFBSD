import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

class FoodborneIllnessPredictionModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.prepare_data()
    
    def prepare_data(self):
        # Data cleaning and preprocessing
        self.data['Date'] = pd.to_datetime(self.data['Year'].astype(str) + '-' + self.data['Month'].astype(str))
        self.data['Season'] = self.data['Date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
    
    def feature_engineering(self):
        # Create risk features
        state_risk = self.data.groupby('State')[['Illnesses', 'Hospitalizations', 'Fatalities']].mean()
        location_risk = self.data.groupby('Location')[['Illnesses', 'Hospitalizations', 'Fatalities']].mean()
        seasonal_risk = self.data.groupby('Season')[['Illnesses', 'Hospitalizations', 'Fatalities']].mean()
        
        return state_risk, location_risk, seasonal_risk
    
    def prepare_ml_dataset(self):
        # Prepare features and target variables
        features = ['State', 'Location', 'Food', 'Ingredient', 'Season']
        categorical_features = ['State', 'Location', 'Food', 'Ingredient', 'Season']
        
        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Multi-output models
        X = self.data[features]
        y_severity = self.data[['Illnesses', 'Hospitalizations', 'Fatalities']]
        
        return X, y_severity, preprocessor
    
    def train_models(self):
        X, y_severity, preprocessor = self.prepare_ml_dataset()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y_severity, test_size=0.2, random_state=42)
        
        # Severity Prediction Model
        severity_model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', MultiOutputRegressor(GradientBoostingRegressor(random_state=42)))
        ])
        
        # Geographic Risk Classification Model
        risk_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train models
        severity_model.fit(X_train, y_train)
        risk_model.fit(X_train, y_train['Illnesses'] > y_train['Illnesses'].median())
        
        return severity_model, risk_model
    
    def generate_risk_analysis(self):
        state_risk, location_risk, seasonal_risk = self.feature_engineering()
        
        # Comprehensive risk analysis
        risk_analysis = {
            'state_risk': state_risk,
            'location_risk': location_risk,
            'seasonal_risk': seasonal_risk
        }
        
        return risk_analysis
    
    def predict_outbreak(self, input_data):
        severity_model, risk_model = self.train_models()
        
        # Predict severity
        severity_prediction = severity_model.predict(input_data)
        
        # Predict risk level
        risk_prediction = risk_model.predict(input_data)
        
        return {
            'severity': severity_prediction,
            'risk_level': risk_prediction
        }

# Example usage
def main():
    model = FoodborneIllnessPredictionModel('foodborne_illness_data.csv')
    
    # Generate risk analysis
    risk_analysis = model.generate_risk_analysis()
    print("Risk Analysis:")
    for key, value in risk_analysis.items():
        print(f"{key}:\n{value}\n")
    
    # Example prediction input
    sample_input = pd.DataFrame({
        'State': ['California'],
        'Location': ['Restaurant'],
        'Food': ['Seafood'],
        'Ingredient': ['Fish'],
        'Season': ['Summer']
    })
    
    # Make predictions
    predictions = model.predict_outbreak(sample_input)
    print("\nPrediction Results:")
    print("Severity Prediction:", predictions['severity'])
    print("Risk Level:", predictions['risk_level'])

if __name__ == "__main__":
    main()