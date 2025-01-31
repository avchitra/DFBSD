from model import OutbreakPredictionModel
import pandas as pd

def main():
    model = OutbreakPredictionModel('outbreaks.csv')

    if model.clf is None:
        model.create_model_pipeline()
    #evaluation = model.comprehensive_evaluation()
    
    #print("Confusion Matrix:\n", evaluation['Confusion_Matrix'])
    #print("\nClassification Report:\n", evaluation['Classification_Report'])
    #print(f"\nROC AUC Score: {evaluation['ROC_AUC_Score']:.4f}")
    #print(f"Average Precision: {evaluation['Average_Precision']:.4f}")
    
    # Example of predicting outbreak risk for new data
    new_data = pd.DataFrame({
        'State': ['California'],
        'Location': ['Restaurant'],
        'Food': ['Seafood'],
        'Ingredient': ['Tuna'],
        'Species': ['Human'],
        'Serotype/Genotype': ['Unknown']
    })
    
    outbreak_risk_prob = model.predict_outbreak_risk(new_data)
    print("\nPredicted Outbreak Risk Probability:")
    print(outbreak_risk_prob)

if __name__ == "__main__":
    main()