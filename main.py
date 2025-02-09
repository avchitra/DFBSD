import pandas as pd
from pathlib import Path
import pickle

def load_data(file_path):
    """Load and prepare the dataset"""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find data file at {file_path}")
        return None

def load_model_from_file(model_path):
    """Load the trained models from the specified file"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            return model_data
    except FileNotFoundError:
        print(f"Error: Could not find model file at {model_path}")
        return None

def main():
    """
    Main function to load pre-trained models and make predictions on sample data.
    If no model exists, trains a new one using the provided dataset.
    """
    # Define paths
    MODEL_PATH = "models/dual_model.pkl"
    DATA_PATH = "data/outbreaks.csv"
    
    # Initialize DualModelSystem
    from model import DualModelSystem
    dual_system = DualModelSystem()

    # Try to load existing models
    model_data = load_model_from_file(MODEL_PATH)
    if model_data is None:
        print("Training new models...")
        
        # Load and prepare training data
        df = load_data(DATA_PATH)
        if df is None:
            return
        
        # Train the Twitter and Outbreak models
        dual_system.train_twitter_model(
            train_path='TWEET-FID/LREC_BSC/train.p',
            dev_path='TWEET-FID/LREC_BSC/dev.p',
            test_path='TWEET-FID/LREC_BSC/test.p'
        )
        dual_system.train_outbreak_model('outbreaks.csv')
        
        # Save the trained models
        dual_system.save_models(MODEL_PATH)
    else:
        # Load the models into the DualModelSystem
        dual_system.load_models(MODEL_PATH)
        print("Loaded existing models...")
    
    # Create sample test data for prediction
    test_data = pd.DataFrame({
        'State': ['California', 'New York'],
        'Location': ['Restaurant', 'Grocery Store'],
        'Food': ['Seafood', 'Produce'],
        'Species': ['Human', 'Human'],
        'Serotype/Genotype': ['Unknown', 'Unknown'],
        'Month': [6, 7]  # June and July
    })
    
    # Preprocess test data for Outbreak Predictor
    X_test = dual_system.outbreak_predictor.preprocess_features(test_data)
    
    # Make predictions using Outbreak Predictor
    outbreak_probabilities = dual_system.outbreak_predictor.model.predict_proba(X_test)
    
    # Preprocess test data for Twitter Classifier (if needed)
    twitter_features = dual_system.twitter_classifier.process_tweet_features(test_data)
    
    # Make predictions using Twitter Classifier (Optional, based on use case)
    twitter_predictions = dual_system.twitter_classifier.model.predict(twitter_features)
    
    # Print results for Outbreak Prediction
    print("\nOutbreak Prediction Results:")
    print("-" * 50)
    for i, (_, row) in enumerate(test_data.iterrows()):
        print(f"\nScenario {i+1}:")
        print(f"Location: {row['State']}, {row['Location']}")
        print(f"Food Type: {row['Food']}")
        print(f"Outbreak Risk Probability: {outbreak_probabilities[i][1]:.2%}")
        
    # Optionally print Twitter classification results
    print("\nTwitter Classification Results:")
    print("-" * 50)
    for i, (_, row) in enumerate(test_data.iterrows()):
        print(f"\nScenario {i+1}:")
        print(f"Location: {row['State']}, {row['Location']}")
        print(f"Food Type: {row['Food']}")
        print(f"Twitter Classification: {'Risk' if twitter_predictions[i] == 1 else 'No Risk'}")

if __name__ == "__main__":
    main()
