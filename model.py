import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class TwitterClassifier:
    """Processes Twitter data and classifies foodborne illness mentions"""
    def __init__(self):
        self.label_encoders = {}
        self.model = None
        
    def load_bsc_data(self, train_path, dev_path, test_path):
        """Load BSC-aggregated Twitter data"""
        # Load data
        self.train_data = pd.read_pickle(train_path)
        self.dev_data = pd.read_pickle(dev_path)
        self.test_data = pd.read_pickle(test_path)
        
        # Print data information
        print("\nTraining Data Info:")
        print("Columns:", self.train_data.columns.tolist())
        print("\nSample data point:")
        print(self.train_data.iloc[0])
        
        # Check for data consistency
        train_cols = set(self.train_data.columns)
        dev_cols = set(self.dev_data.columns)
        test_cols = set(self.test_data.columns)
        
        if train_cols != dev_cols or train_cols != test_cols:
            print("\nWarning: Column mismatch between datasets!")
            print("Train columns:", train_cols)
            print("Dev columns:", dev_cols)
            print("Test columns:", test_cols)
        
        print(f"\nLoaded {len(self.train_data)} training samples")
        print(f"Loaded {len(self.dev_data)} validation samples")
        print(f"Loaded {len(self.test_data)} test samples")
    
    def process_tweet_features(self, data):
        """Process tweet features including entities and relevance"""
        features = pd.DataFrame()
        
        # Get all available columns
        available_cols = data.columns.tolist()
        print("\nAvailable columns for feature extraction:", available_cols)
        
        # Extract entity counts (safely check for column existence)
        if 'entity_label' in available_cols:
            features['food_entities'] = data.apply(lambda x: sum(1 for label in x['entity_label'] if label.startswith('B-FOOD')), axis=1)
            features['location_entities'] = data.apply(lambda x: sum(1 for label in x['entity_label'] if label.startswith('B-LOC')), axis=1)
            features['symptom_entities'] = data.apply(lambda x: sum(1 for label in x['entity_label'] if label.startswith('B-SYMP')), axis=1)
        else:
            print("Warning: 'entity_label' column not found")
            features['food_entities'] = 0
            features['location_entities'] = 0
            features['symptom_entities'] = 0
            
        # If 'relevant_entity_label' exists, use it
        if 'relevant_entity_label' in available_cols:
            features['relevant_entities'] = data.apply(lambda x: sum(1 for label in x['relevant_entity_label'] if label != 'O'), axis=1)
        else:
            print("Warning: 'relevant_entity_label' column not found")
            features['relevant_entities'] = 0
            
        return features
    
    def prepare_data(self):
        """Prepare features and labels for training"""
        # Process features
        self.X_train = self.process_tweet_features(self.train_data)
        self.X_dev = self.process_tweet_features(self.dev_data)
        self.X_test = self.process_tweet_features(self.test_data)
        
        # Get labels (sentence classification)
        if 'sentence_class' not in self.train_data.columns:
            raise KeyError("'sentence_class' column not found in the data")
            
        self.y_train = self.train_data['sentence_class']
        self.y_dev = self.dev_data['sentence_class']
        self.y_test = self.test_data['sentence_class']
        
        print("\nFeature shapes:")
        print(f"X_train: {self.X_train.shape}")
        print(f"X_dev: {self.X_dev.shape}")
        print(f"X_test: {self.X_test.shape}")


class OutbreakPredictor:
    """Predicts outbreak risk using CDC data and Twitter signals"""
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,  # Limit tree depth to prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced'
        )
        self.label_encoders = {}
        self.feature_columns = None
        self.cdc_data = None
        self.month_map = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }

    def load_cdc_data(self, filepath):
        """Load CDC outbreak data"""
        self.cdc_data = pd.read_csv(filepath)
        print(f"Loaded {len(self.cdc_data)} CDC outbreak records")

    def convert_month_to_number(self, month_str):
        """Convert month string to number (1-12)"""
        if pd.isna(month_str):
            return 1
        try:
            month_num = int(month_str)
            if 1 <= month_num <= 12:
                return month_num
        except (ValueError, TypeError):
            pass
        month_str = str(month_str).lower().strip()
        return self.month_map.get(month_str, 1)

    def preprocess_features(self, df):
        """Preprocess CDC data features"""
        X = df.copy()
        
        # Remove any target-related columns to prevent leakage
        leakage_cols = ['Illnesses', 'Hospitalized', 'Deaths']
        for col in leakage_cols:
            if col in X.columns:
                X.drop(col, axis=1, inplace=True)
        
        # Handle categorical variables (with consistency checks)
        categorical_cols = ['State', 'Location', 'Food', 'Status']
        for col in categorical_cols:
            if col in X.columns:
                encoded_col = f'{col}_encoded'
                # Check category counts
                value_counts = X[col].value_counts()
                if len(value_counts) < 2:
                    print(f"Warning: Column {col} has low variance")
                    continue
                    
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[encoded_col] = self.label_encoders[col].fit_transform(X[col].fillna('Unknown'))
                else:
                    X[col] = X[col].fillna('Unknown')
                    known_categories = set(self.label_encoders[col].classes_)
                    X[col] = X[col].apply(lambda x: 'Unknown' if x not in known_categories else x)
                    X[encoded_col] = self.label_encoders[col].transform(X[col])
        
        # Convert Month to cyclical features
        if 'Month' in X.columns:
            month_numbers = X['Month'].apply(self.convert_month_to_number)
            X['Month_sin'] = np.sin(2 * np.pi * month_numbers / 12)
            X['Month_cos'] = np.cos(2 * np.pi * month_numbers / 12)
            X.drop('Month', axis=1, inplace=True)

        # Add derived features
        if 'Year' in X.columns:
            X['Year'] = pd.to_numeric(X['Year'], errors='coerce')
            current_year = pd.Timestamp.now().year
            X['Years_Ago'] = current_year - X['Year']
            X['Year'] = X['Year'].fillna(X['Year'].median())
        
        # Drop any remaining non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Update feature columns
        self.feature_columns = X.columns.tolist()
        
        return X

    def create_target(self, df, threshold=10):
        """Create binary target for significant outbreaks"""
        # Use multiple factors to determine outbreak severity
        has_illnesses = df['Illnesses'].fillna(0) >= threshold
        has_hospitalizations = df['Hospitalizations'].fillna(0) >= threshold/5  # 20% hospitalization rate
        has_fatalities = df['Fatalities'].fillna(0) > 0
        
        return (has_illnesses | has_hospitalizations | has_fatalities).astype(int)

class DualModelSystem:
    """Combines Twitter and CDC data for comprehensive outbreak prediction"""
    def __init__(self):
        self.twitter_classifier = TwitterClassifier()
        self.outbreak_predictor = OutbreakPredictor()
        
    def train_twitter_model(self, train_path, dev_path, test_path):
        """Train the Twitter classification model"""
        self.twitter_classifier.load_bsc_data(train_path, dev_path, test_path)
        self.twitter_classifier.prepare_data()
        
        # Train classifier with cross-validation
        self.twitter_classifier.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            class_weight='balanced'
        )
        
        self.twitter_classifier.model.fit(
            self.twitter_classifier.X_train,
            self.twitter_classifier.y_train
        )
        
        # Evaluate model performance on the development set
        # The dev set is used to tune hyperparameters and evaluate model generalization
        # y_pred = self.twitter_classifier.model.predict(self.twitter_classifier.X_dev)
        # print("\nTwitter Model Evaluation (Development Set):")
        # print(classification_report(self.twitter_classifier.y_dev, y_pred))
        
        # Additional evaluation on test set
        y_test_pred = self.twitter_classifier.model.predict(self.twitter_classifier.X_test)
        print("\nTwitter Model Evaluation (Test Set):")
        print(classification_report(self.twitter_classifier.y_test, y_test_pred))
        
        y_test_pred_proba = self.twitter_classifier.model.predict_proba(self.twitter_classifier.X_test)[:,1]
        print(f"\nTwitter Model AUC-ROC Score (Test Set): {roc_auc_score(self.twitter_classifier.y_test, y_test_pred_proba):.3f}")
        print("\nTwitter Model Confusion Matrix (Test Set):")
        print(confusion_matrix(self.twitter_classifier.y_test, y_test_pred))

    def train_outbreak_model(self, cdc_data_path):
        """Train the outbreak prediction model"""
        self.outbreak_predictor.load_cdc_data(cdc_data_path)
        
        # Prepare features and target
        X = self.outbreak_predictor.preprocess_features(self.outbreak_predictor.cdc_data)
        y = self.outbreak_predictor.create_target(self.outbreak_predictor.cdc_data)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Further split training data to create validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Train model
        self.outbreak_predictor.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        # y_val_pred = self.outbreak_predictor.model.predict(X_val)
        # print("\nOutbreak Model Evaluation (Validation Set):")
        # print(classification_report(y_val, y_val_pred))
        
        # Evaluate on test set
        y_test_pred = self.outbreak_predictor.model.predict(X_test)
        print("\nOutbreak Model Evaluation (Test Set):")
        print(classification_report(y_test, y_test_pred))

        from sklearn.metrics import roc_auc_score, confusion_matrix

        # Calculate and print AUC-ROC score
        roc_auc = roc_auc_score(y_test, y_test_pred)
        print("\nAUC-ROC Score (Test Set):", roc_auc)

        # Calculate and print confusion matrix
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        print("\nConfusion Matrix (Test Set):\n", conf_matrix)

        
        # Print feature importances
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': self.outbreak_predictor.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature Importances:")
        print(importances)

    def save_models(self, path):
        """Save both models"""
        model_data = {
            'twitter_model': self.twitter_classifier.model,
            'outbreak_model': self.outbreak_predictor.model,
            'twitter_encoders': self.twitter_classifier.label_encoders,
            'outbreak_encoders': self.outbreak_predictor.label_encoders,
            'feature_columns': self.outbreak_predictor.feature_columns
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nModels saved to: {path}")
        
    def load_models(self, path):
        """Load both models"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            self.twitter_classifier.model = model_data['twitter_model']
            self.outbreak_predictor.model = model_data['outbreak_model']
            self.twitter_classifier.label_encoders = model_data['twitter_encoders']
            self.outbreak_predictor.label_encoders = model_data['outbreak_encoders']
            self.outbreak_predictor.feature_columns = model_data['feature_columns']
        print(f"\nModels loaded from: {path}")

if __name__ == "__main__":
    # Initialize the dual model system
    system = DualModelSystem()
    
    # Train Twitter model
    system.train_twitter_model(
        train_path='TWEET-FID/LREC_BSC/train.p',
        dev_path='TWEET-FID/LREC_BSC/dev.p',
        test_path='TWEET-FID/LREC_BSC/test.p'
    )
    
    # Train outbreak model
    system.train_outbreak_model('outbreaks.csv')

    # Save models
    system.save_models('models/dual_model.pkl')