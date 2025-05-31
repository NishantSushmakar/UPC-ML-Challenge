import pandas as pd 
import config
import numpy as np
from feature_creation import *
import joblib 
from sklearn.pipeline import Pipeline
from training import zero_one_loss

def test_prediction_group_models(test_df):

    # 1. Load group models
    group_models = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, 'group_models.pkl'))
    
    # 2. Create feature pipeline (same as training)
    feature_pipeline = Pipeline(steps=[
        ("Language Features", LanguageFeature()),
        ("Structural Language Groups", StructuralLanguageGrouper(n_clusters=5)),
        ("Graph Features", GraphFeatures()),
        ("Node Features", NodeFeatures()),
        ("Dataset Creation", FormatDataFrame()),
        ("Language One Hot Encoding", LanguageOHE(
            enc_lan="temp_lan_encoder.pkl",
            enc_lan_family="temp_lan_family_encoder.pkl",
            enc_structural_group="temp_structural_group_encoder.pkl"))  
    ])
    
    # 3. Process test data
    test_data = feature_pipeline.transform(test_df)
    
    # 4. Define columns needed for prediction (same as training)
    cols_to_use = [
        "eccentricity", "closeness_cent", "subgraph_cent", "betweeness_cent",
        "page_cent", "number_of_nodes", "num_leaf_neighbors",
        "is_leaf", "eigen_cent", "degree", "vote_rank_score", "participation_diameter"
    ] + [col for col in test_data.columns 
        if (col.startswith('language_') and not col.startswith('language_group') )]  # Will only include existing columns
    
    # 5. Make group-specific predictions
    all_predictions = []
    for group_name, model in group_models.items():
        group_data = test_data[test_data['structural_group'] == group_name]
        if len(group_data) > 0:
            print(f"Predicting for {group_name} ({len(group_data)} samples)")
            X_group = group_data[cols_to_use]
            group_data['prediction_probability'] = model.predict_proba(X_group)[:, 1]
            all_predictions.append(group_data)
    
    # 6. Combine all predictions
    combined_preds = pd.concat(all_predictions)
    
    # 7. Select root for each sentence (highest probability)
    test_max_rows = combined_preds.loc[combined_preds.groupby(
        ['id', 'sentence', 'language'])['prediction_probability'].idxmax()]
    
    # 8. Format results
    test_result = test_max_rows[['id', 'sentence', 'language', 'node_number']] \
        .rename(columns={'node_number': 'root'})
    
    # 9. Merge with original test data
    final_df = pd.merge(test_df, test_result, on=['id', 'sentence', 'language'], how='inner') \
                [['id', 'root']].sort_values(by='id')
    
    # 10. Save predictions
    final_df.to_csv(os.path.join(config.DATA_PATH, 'group_models_submission.csv'), index=False)
    
    # 11. Evaluate if true labels available
    if os.path.exists(config.TEST_TRUE_DATA_PATH):
        test_true_df = pd.read_csv(config.TEST_TRUE_DATA_PATH)
        test_zero_one_loss = zero_one_loss(test_true_df['root'], final_df['root'])
        print(f"\nTest Set Accuracy: {1 - test_zero_one_loss:.4f}")
        
        # Per-language metrics
        merged = test_true_df.merge(final_df, on='id', suffixes=('_true', '_pred'))
        merged = merged.merge(test_data[['id', 'language']].drop_duplicates(), on='id')
        
        lang_scores = merged.groupby('language').apply(
            lambda x: 1 - zero_one_loss(x['root_true'], x['root_pred'])
        ).reset_index(name='accuracy')
        
        print("\nPer-Language Accuracy:")
        print(lang_scores.sort_values('accuracy', ascending=False).to_string(index=False))
    
    return final_df

if __name__ == "__main__":
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    results = test_prediction_group_models(test_df)