import pandas as pd 
import config
import numpy as np
from feature_creation import *
import joblib 


def test_prediction(test_df, best_model, best_fold):
    # Create feature pipeline with exact same steps as training
    feature_pipeline = Pipeline(steps=[
        ("Language Features", LanguageFeature()),
        ("Graph Features", GraphFeatures()),
        ("Node Features", NodeFeatures()),
        ("Dataset Creation", FormatDataFrame()),
        ("Language One Hot Encoding", LanguageOHE(
            enc_lan=f"{best_model}/lan_encoder_{best_model}_stratified.pkl",
            enc_lan_family=f"{best_model}/lan_family_encoder_{best_model}_stratified.pkl"
        ))
    ])

    # Transform test data
    test_data_start = feature_pipeline.transform(test_df)
    
    # Drop columns exactly as done in training
    test_data = test_data_start.drop(columns=config.TEST_DROP_COLS)
    
    # Load and apply model
    clf = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, f'{best_model}/{best_model}_{best_fold}.pkl'))
    
    # Get predictions
    test_pred_proba = clf.predict_proba(test_data)[:, 1]
    test_data_start['prediction_probability'] = test_pred_proba

    # Get max probability predictions for each sentence
    test_max_rows = test_data_start.loc[test_data_start.groupby(by=['id', 'sentence', 'language'])['prediction_probability'].idxmax()]
    test_result = test_max_rows[['id', 'sentence', 'language', 'node_number']]
    test_result = test_result.rename(columns={'node_number': 'root'})

    # Merge and prepare final submission
    test_df = pd.merge(test_df, test_result, on=['id', 'sentence', 'language'], how='inner')
    test_df = test_df[['id', 'root']].sort_values(by='id')

    # Save submission
    test_df.to_csv(os.path.join(config.DATA_PATH, f'{best_model}_{best_fold}_submission.csv'), index=False)


if __name__ == "__main__":

    test_df = pd.read_csv(config.TEST_DATA_PATH)
    best_model = 'lgbm_logloss'
    best_fold = 7
    test_prediction(test_df,best_model,best_fold)









