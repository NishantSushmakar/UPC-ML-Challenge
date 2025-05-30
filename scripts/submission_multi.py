import pandas as pd 
import config
import numpy as np
from feature_creation import *
import joblib 
import os

def test_prediction_multi(test_df, model_name, japanese_model_name, best_fold):
    """
    Make predictions using separate models for Japanese and non-Japanese languages.
    
    Parameters:
    - test_df: DataFrame containing test data
    - model_name: Model name for non-Japanese languages
    - japanese_model_name: Model name for Japanese language
    - best_fold: Fold number to use for prediction
    """
    # Create feature pipeline
    feature_pipeline = Pipeline(steps=[
        ("Language Features", LanguageFeature()),
        ("Graph Features", GraphFeatures()),
        ("Node Features", NodeFeatures()),
        ("Dataset Creation", FormatDataFrame()),
        ("Language One Hot Encoding", LanguageOHE(
            enc_lan=f"{model_name}/lan_encoder_{model_name}_{best_fold}.pkl",
            enc_lan_family=f"{model_name}/lan_family_encoder_{model_name}_{best_fold}.pkl"
        ))
    ])
    
    # Process test data through feature pipeline
    test_data = feature_pipeline.transform(test_df)
    
    # Split data for Japanese and non-Japanese
    jp_test = test_data[test_data.language == 'Japanese'].reset_index(drop=True)
    other_test = test_data[test_data.language != 'Japanese'].reset_index(drop=True)
    
    print(f"Number of Japanese samples: {len(jp_test)}")
    print(f"Number of non-Japanese samples: {len(other_test)}")
    
    def predict_with_model(data, model_name, is_japanese=False):
        if data.empty:
            return None
        
        # Prepare features
        x_test = data.drop(columns=config.TEST_DROP_COLS)
        
        # Load model
        model_path = f"{model_name}/{model_name}_{'jp' if is_japanese else 'other'}_{best_fold}.pkl"
        print(f"Loading model from: {model_path}")
        clf = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, model_path))
        
        # Get predictions
        test_pred_proba = clf.predict_proba(x_test)[:, 1]
        
        # Add predictions to the dataframe
        data['prediction_probability'] = test_pred_proba
        return data
    
    # Get predictions for both Japanese and non-Japanese data
    jp_test_data = predict_with_model(jp_test, japanese_model_name, is_japanese=True)
    other_test_data = predict_with_model(other_test, model_name, is_japanese=False)
    
    # Combine predictions while preserving order
    def combine_predictions(data1, data2):
        if data1 is None:
            return data2
        if data2 is None:
            return data1
        
        # Combine while preserving the original order
        combined = pd.concat([data1, data2])
        # Sort by id, sentence and language to ensure consistent ordering
        combined = combined.sort_values(['id', 'sentence', 'language'])
        return combined
    
    # Combine test data
    test_data = combine_predictions(jp_test_data, other_test_data)
    
    # Get max probability predictions for each sentence
    test_max_rows = test_data.loc[test_data.groupby(by=['id', 'sentence', 'language'])['prediction_probability'].idxmax()]
    test_result = test_max_rows[['id', 'sentence', 'language', 'node_number']]
    test_result = test_result.rename(columns={'node_number': 'root'})

    print(test_result.shape)
    print(test_df.shape)
    
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
    submission_path = os.path.join(config.DATA_PATH, f'{model_name}_{japanese_model_name}_{best_fold}_submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    print(f"Final number of predictions: {len(submission_df)}")
    print(f"Final number of unique IDs: {len(submission_df['id'].unique())}")

if __name__ == "__main__":
    # Load test data
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    
    # Set model names and fold
    model_name = 'lgbm_other'
    japanese_model_name = 'lgbm_jp'
    best_fold = 0
    
    # Generate predictions
    test_prediction_multi(test_df, model_name, japanese_model_name, best_fold) 