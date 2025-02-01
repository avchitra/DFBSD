from model import OutbreakPredictionModel
import pandas as pd

def main():
    """
    Main function to initialize the OutbreakPredictionModel, create the model pipeline 
    if necessary, and predict outbreak risk for sample test data.

    - Loads outbreak data from 'outbreaks.csv'.
    - Checks if a pre-trained model exists; if not, trains a new model.
    - Uses sample test data to predict the risk of a foodborne illness outbreak.
    - Prints the predicted outbreak risk probabilities for the test data.
    """

    model = OutbreakPredictionModel('outbreaks.csv')

    # If the model hasn't been trained yet, check if a pre-trained model exists; if not, train a new model.
    #if model.clf is None:
    #    model.create_model_pipeline()
    evaluation = model.evaluate_model()
    
    print("Confusion Matrix:\n", evaluation['Confusion_Matrix'])
    print("\nClassification Report:\n", evaluation['Classification_Report'])
    print(f"\nROC AUC Score: {evaluation['ROC_AUC_Score']:.4f}")
    print(f"Average Precision: {evaluation['Average_Precision']:.4f}")
    
    #test_data = pd.DataFrame({
   #     'State': ['California'],
    #    'Location': ['Restaurant'],
    #    'Food': ['Seafood'],
    #    'Ingredient': ['Tuna'],
    #    'Species': ['Human'],
    #    'Serotype/Genotype': ['Unknown']
    #})
    
    #outbreak_risk_prob = model.predict_outbreak_risk(test_data)
    #print("\nPredicted Outbreak Risk Probability:")
    #print(outbreak_risk_prob)

if __name__ == "__main__":
    main()