import pandas as pd 
import config
import numpy as np
from feature_creation import *
import joblib 
import os

def test_prediction_ensemble(test_df, model_configs, best_fold):
    """
    Make predictions using an ensemble of models with their respective weights.
    
    Parameters:
    - test_df: DataFrame containing test data
    - model_configs: List of dictionaries containing model configurations
        Each dict should have:
        - model_name: Name of the model
        - weight: Weight to give to this model's predictions (default: 1.0)
    - best_fold: Fold number to use for prediction
    """
    # Create feature pipeline using the first model's configuration
    feature_pipeline = Pipeline(steps=[
        ("Language Features", LanguageFeature()),
        ("Graph Features", GraphFeatures()),
        ("Node Features", NodeFeatures()),
        ("Dataset Creation", FormatDataFrame()),
        ("Language One Hot Encoding", LanguageOHE(
            enc_lan=f"{model_configs[0]['model_name']}/lan_encoder_{model_configs[0]['model_name']}_{best_fold}.pkl",
            enc_lan_family=f"{model_configs[0]['model_name']}/lan_family_encoder_{model_configs[0]['model_name']}_{best_fold}.pkl"
        ))
    ])
    
    # Process test data through feature pipeline
    test_data = feature_pipeline.transform(test_df)
    
    # Initialize DataFrame to store combined predictions
    test_combined = None
    
    # Get predictions from each model
    for m_config in model_configs:
        model_name = m_config['model_name']
        weight = m_config.get('weight', 1.0)
        
        print(f"\nProcessing model: {model_name} (weight: {weight})")
        
        # Prepare features
        x_test = test_data.drop(columns=config.TEST_DROP_COLS)
        
        # Load model
        model_path = f"{model_name}/{model_name}_{best_fold}.pkl"
        print(f"Loading model from: {model_path}")
        clf = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, model_path))
        
        # Get predictions
        test_pred_proba = clf.predict_proba(x_test)[:, 1] * weight
        
        # Store predictions
        if test_combined is None:
            test_combined = test_data.copy()
            test_combined['prediction_probability'] = test_pred_proba
        else:
            test_combined['prediction_probability'] += test_pred_proba
    
    # Get max probability predictions for each sentence
    test_max_rows = test_combined.loc[test_combined.groupby(by=['id', 'sentence', 'language'])['prediction_probability'].idxmax()]
    test_result = test_max_rows[['id', 'sentence', 'language', 'node_number']]
    test_result = test_result.rename(columns={'node_number': 'root'})
    
    # Merge predictions back to original dataframe
    test_df = pd.merge(test_df, test_result, on=['id', 'sentence', 'language'], how='inner')
    
    # Prepare final submission - ensure one prediction per id
    submission_df = test_df[['id', 'root']].drop_duplicates(subset=['id']).sort_values(by='id')
    
    # Verify we have exactly one prediction per id
    if len(submission_df) != len(submission_df['id'].unique()):
        print("\nWarning: Multiple predictions found for some IDs")
        print("Sample of duplicate IDs:")
        duplicates = submission_df[submission_df.duplicated(subset=['id'], keep=False)]
        print(duplicates.head())
        # Keep only the first prediction for each id
        submission_df = submission_df.drop_duplicates(subset=['id'], keep='first')
    
    # Save submission
    model_names = '_'.join([c['model_name'] for c in model_configs])
    submission_path = os.path.join(config.DATA_PATH, f'ensemble_{model_names}_{best_fold}_submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    print(f"Final number of predictions: {len(submission_df)}")
    print(f"Final number of unique IDs: {len(submission_df['id'].unique())}")

if __name__ == "__main__":
    # Load test data
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    
    # Define model configurations
    model_configs = [
        {'model_name': 'lgbm', 'weight': 1.0},
        {'model_name': 'lda', 'weight': 1.0},
        {'model_name': 'xgb', 'weight': 1.0},
        {'model_name': 'mnb', 'weight': 1.0}
    ]
    
    best_fold = 1
    
    # Generate predictions using ensemble
    test_prediction_ensemble(test_df, model_configs, best_fold)