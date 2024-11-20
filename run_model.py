import numpy as np
from morphological_analysis import MorphologicalPredictor
import json
import os

def main():
    print("\nInitializing Morphological Predictor...")
    predictor = MorphologicalPredictor()
    
    print("\nTraining model with cross-validation...")
    predictor.train_with_cross_validation("CatA_Simple", n_folds=10)
    
    test_file = "combined_output.json"
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            test_cases = json.load(f)
            
        for i, test_case in enumerate(test_cases):
            input_matrix = np.array(test_case['input'])
            prediction = predictor.predict(input_matrix)
            print(f"\nTest case {i + 1}:")
            print(f"Prediction: {'All pixels change' if prediction else 'Not all pixels change'}")

if __name__ == "__main__":
    main()