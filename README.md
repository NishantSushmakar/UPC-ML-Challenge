# UPC-ML-Challenge

This repository contains the code for the UPC-ML Challenge. This work focuses on identifying the root node in syntactic dependency trees, which are modeled as
free trees. The task is framed as a binary classification problem where each node in a tree is labeled as either root (1) or non-root (0). Prediction is based on centrality measures that quantify the structural importance of nodes, complemented by additional computed features.

**Implemented by:** Nishant Sushmakar and Marwah Sulaiman. 

**Supervised by:** Prof. Marta Arias

## Getting Started

1. Clone this repository and navigate to it:
   ```bash
   git clone https://github.com/NishantSushmakar/UPC-ML-Challenge.git
   cd UPC-ML-Challenge

## Training the Model

1. The training data (`train.csv`) is located in the `data` directory. To use a different training set, replace it with your own file under the same name.

2. Navigate to the `scripts` directory:
   ```bash
   cd scripts

3. Run the `training.py` script to train the model. You can choose which model to train by uncommenting the corresponding line near the end of the script (e.g., `'lr'` for Logistic Regression, `'lgbm_zero_one_loss'` for LightGBM with zero-one loss, `'xgb_logloss'` for XGBoost with log loss). The current uncommented one is our best-performing model.

   To train the model:

   ```bash
   python training.py
   ```
   
   **Note: Random Forest Models**  
      To train Random Forest variants:
      
      ```bash
      cd scripts/RF
      python training.py
      ```

4. The cross-validation results will be displayed in the terminal and saved to a file in the `results` directory.


## Testing the Model

1. The test data (`test.csv`) is located in the `data` directory. Replace it with your own test set if needed.

2. In the `submission.py` file:
   - Specify the model you want to test (e.g., `'lgbm_logloss'`)
   - Specify the best-performing fold number (from your training results)
   - Example configuration at the end of the file:
     ```python
     best_model = 'lgbm_logloss'
     best_fold = 7 
     ```

3. In the same `scripts` directory, run the submission script:
   ```bash
   python submission.py
   
4. The script will generate a prediction file with the following naming format, and it has the predicted root node for each sentence id.
   ```bash
   {best_model}_{best_fold}_submission.csv

## Notes

To run the isomorphism experiments:

1. Navigate to the Isomorphism directory:
   ```bash
   cd scripts/Isomorphism
   ```
2. Run the isomorphism script:
   ```bash
   python script_Isomorphism.py
   ```
