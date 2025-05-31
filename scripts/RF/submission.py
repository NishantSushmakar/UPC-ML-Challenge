import pandas as pd 
import config
import numpy as np
from feature_creation import *
from training import *

import joblib 


def test_prediction(test_df,best_model, best_fold):

    feature_pipeline = Pipeline(steps=[
                ("Language Features",LanguageFeature()),
                ("Graph Features",GraphFeatures()),
                ("Node Features",NodeFeatures()),
                ("Dataset Creation",FormatDataFrame()),
                ("Language One Hot Encoding",LanguageOHE(enc_lan="temp_lan_encoder.pkl", enc_lan_family="temp_lan_family_encoder.pkl"))  
            ])
    


    test_data_start = feature_pipeline.transform(test_df)
    # test_data = test_data_start.drop(columns=config.TEST_DROP_COLS)



    cols_to_use = [ 
        "eccentricity",
    "closeness_cent",
    "subgraph_cent",
    "betweeness_cent",
    # "current_flow_closeness",
    # "degree_cent",
    "page_cent",
    "number_of_nodes",
    "num_leaf_neighbors",
    "is_leaf",
    "eigen_cent",

    # try with these
    "degree",
    # "vote_rank_score",
    # "largest_component_removed",
    # "participation_diameter",

    # ]
    'language_Arabic',
    'language_Chinese',
    'language_Czech',
    'language_English',
    'language_Finnish',
    'language_French',
    'language_Galician',
    'language_German',
    'language_Hindi',
    'language_Icelandic',
    'language_Indonesian',
    'language_Italian',
    'language_Japanese',
    'language_Korean',
    'language_Polish',
    'language_Portuguese',
    'language_Russian',
    'language_Spanish',
    'language_Swedish',
    'language_Thai',
    'language_Turkish']

    
    # x_train_data = train_data.drop(columns=config.TRAIN_DROP_COLS)
    test_data = test_data_start[cols_to_use]


    clf = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,f'{best_model}/{best_model}_{best_fold}.pkl'))

    test_pred_proba = clf.predict_proba(test_data)[:,1]
    test_data_start['prediction_probability'] = test_pred_proba 

    test_data_start.to_csv("test_data_predictions.csv")

    test_max_rows = test_data_start.loc[test_data_start.groupby(by = ['id','sentence','language'])['prediction_probability'].idxmax()]
    test_result = test_max_rows[['id','sentence','language','node_number']]
    test_result = test_result.rename(columns={'node_number':'root'})

    test_df = pd.merge(test_df,test_result,on=['id','sentence','language'],how='inner')

    test_df = test_df[['id','root']].sort_values(by='id')

    test_df.to_csv(os.path.join(config.DATA_PATH,f'{best_model}_{best_fold}_submission.csv'),index=False)
    
    test_true_df = pd.read_csv(config.TEST_TRUE_DATA_PATH)

    train_zero_one_loss = zero_one_loss(test_true_df['root'],test_df['root'])    

    print(f"\nScore of Test Set {1 - train_zero_one_loss}\n")

    # Merge predictions and true labels
    test_df = test_df.rename(columns={'root': 'root_pred'})
    test_true_df = test_true_df.rename(columns={'root': 'root_true'})

    merged_df = pd.merge(test_true_df, test_df, on='id')
    merged_df = pd.merge(merged_df, test_data_start[['id', 'language']].drop_duplicates(), on='id')

    # Compute per-language accuracy
    per_language_scores = merged_df.groupby('language').apply(
        lambda df: 1 - zero_one_loss(df['root_true'], df['root_pred'])
    ).reset_index(name='accuracy')

    print("\nScore Per Language:")
    print(per_language_scores.sort_values(by='accuracy', ascending=False).to_string(index=False))



if __name__ == "__main__":

    test_df = pd.read_csv(config.TEST_DATA_PATH)
    best_model = 'rff'
    best_fold = 4
    test_prediction(test_df,best_model,best_fold)









