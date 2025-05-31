import pandas as pd 
import config
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,log_loss
import numpy as np
from feature_creation import *
from model_dispatcher import models
from sklearn.preprocessing import MinMaxScaler
import sys
import os
from datetime import datetime
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from imblearn.over_sampling import RandomOverSampler
from collections import defaultdict
from sklearn.pipeline import Pipeline

def zero_one_loss(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of true and predicted labels must match.")
    incorrect = sum(yt != yp for yt, yp in zip(y_true, y_pred))
    return incorrect / len(y_true)

def create_feature_pipeline():
    return Pipeline(steps=[
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

def preprocess_data(df):
    """Process all data through the feature pipeline first"""
    pipeline = create_feature_pipeline()
    return pipeline.fit_transform(df)

def create_groupkfolds(processed_df, orig_group_df, n_folds, group_col):
    """Create folds from already processed data"""
    original_df = orig_group_df.copy()
    expanded_df = processed_df.copy()
    
    expanded_df['kfold'] = -1
    y = expanded_df.is_root
    groups = expanded_df[group_col]

    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for f, (t_, v_) in enumerate(gkf.split(X=expanded_df, y=y, groups=groups)):
        expanded_df.loc[v_, 'kfold'] = f
    
    fold_assignments = expanded_df[['sentence', 'language', 'kfold']].drop_duplicates()
    original_df = original_df.merge(fold_assignments, on=['sentence', 'language'], how='left')
    
    return original_df, expanded_df

def run_group_fold(original_df, expanded_df, fold, model_name, group_name):
    """Run training for a single fold"""
    orig_train = original_df[original_df.kfold != fold].reset_index(drop=True)
    orig_valid = original_df[original_df.kfold == fold].reset_index(drop=True)
    
    train_data = expanded_df[expanded_df.kfold != fold].reset_index(drop=True)
    valid_data = expanded_df[expanded_df.kfold == fold].reset_index(drop=True)

    os.makedirs(f'{config.ONE_HOT_ENCODER_LANGUAGE}/{group_name}', exist_ok=True)

    cols_to_use = [
        "eccentricity", "closeness_cent", "subgraph_cent", "betweeness_cent",
        "page_cent", "number_of_nodes", "num_leaf_neighbors",
        "is_leaf", "eigen_cent", "degree", "vote_rank_score", "participation_diameter"
    ] + [col for col in train_data.columns 
        if (col.startswith('language_') and not col.startswith('language_group'))]
    
    x_train = train_data[cols_to_use + ['sentence', 'language']]
    y_train = train_data.is_root.values
    x_valid = valid_data[cols_to_use]
    y_valid = valid_data.is_root.values

    clf = models[model_name]
    clf.fit(x_train, y_train)
    
    model_path = os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, f'{group_name}/{model_name}_{fold}.pkl')
    joblib.dump(clf, model_path)


    y_train_pred = clf.predict(x_train[cols_to_use])
    y_valid_pred = clf.predict(x_valid)
    
    train_data['prediction_probability'] = clf.predict_proba(x_train[cols_to_use])[:,1]
    valid_data['prediction_probability'] = clf.predict_proba(x_valid)[:,1]

    # Process predictions to ensure one root per sentence
    def set_max_prob_as_root(group):
        group['adjusted_pred'] = 0
        group.loc[group['prediction_probability'].idxmax(), 'adjusted_pred'] = 1
        return group
    
    train_data = train_data.groupby(['sentence', 'language'], group_keys=False).apply(set_max_prob_as_root)
    valid_data = valid_data.groupby(['sentence', 'language'], group_keys=False).apply(set_max_prob_as_root)
    
    # Calculate metrics
    train_result = train_data.loc[train_data.groupby(['sentence','language'])['prediction_probability'].idxmax()]
    valid_result = valid_data.loc[valid_data.groupby(['sentence','language'])['prediction_probability'].idxmax()]
    
    df_train_final = pd.merge(orig_train, train_result[['sentence','language','node_number']].rename(columns={'node_number':'predicted_root'}), on=['sentence','language'])
    df_valid_final = pd.merge(orig_valid, valid_result[['sentence','language','node_number']].rename(columns={'node_number':'predicted_root'}), on=['sentence','language'])

    metrics = {
        'train_f1': f1_score(train_data['adjusted_pred'], y_train_pred, average='macro'),
        'valid_f1': f1_score(valid_data['adjusted_pred'], y_valid_pred, average='macro'),
        'valid_zol': zero_one_loss(df_valid_final['root'], df_valid_final['predicted_root']),
        'valid_auc': roc_auc_score(y_valid, valid_data['prediction_probability'])
    }
    
    return metrics



def train_group_model(group_name, languages, df, processed_df, n_folds=5, model_name='rff'):
    """Train model for a specific group"""
    print(f"\n=== Training {model_name} for {group_name} ===")
    print(f"Languages: {', '.join(languages)}")
    
    group_df = processed_df[processed_df['language'].isin(languages)].copy().reset_index(drop=True)
    orig_group_df = df[df['language'].isin(languages)].copy().reset_index(drop=True)
    
    original_df, expanded_df = create_groupkfolds(group_df, orig_group_df, n_folds, 'sentence')
    
    fold_metrics = defaultdict(list)
    for fold in range(n_folds):
        metrics = run_group_fold(original_df, expanded_df, fold, model_name, group_name)
        for k, v in metrics.items():
            fold_metrics[k].append(v)
        print(f"Fold {fold}: Valid F1 = {metrics['valid_f1']:.3f}, Valid ZOL = {metrics['valid_zol']:.3f}")
    
    # Select best model
    best_fold = np.argmin(fold_metrics['valid_zol'])
    best_model_path = os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, f'{group_name}/{model_name}_{best_fold}.pkl')
    best_model = joblib.load(best_model_path)
    
    print(f"\nBest model for {group_name} - Fold {best_fold}:")
    print(f"Validation F1: {fold_metrics['valid_f1'][best_fold]:.3f}")
    print(f"Validation AUC: {fold_metrics['valid_auc'][best_fold]:.3f}")
    print(f"Validation ZOL: {fold_metrics['valid_zol'][best_fold]:.3f}")
    
    return best_model

def train_structural_group_models(model_name='rff', n_folds=5):
    """Main training function"""
    # Load and preprocess all data first
    df = pd.read_csv(config.TRAINING_DATA_PATH)
    processed_df = preprocess_data(df)
    
    # Get group assignments from the processed data
    group_mapping = defaultdict(list)
    for lang, group in processed_df[['language', 'structural_group']].drop_duplicates().values:
        group_mapping[group].append(lang)
    
    # Train separate models for each group
    group_models = {}
    for group_name, languages in group_mapping.items():
        group_models[group_name] = train_group_model(
            group_name, languages, df, processed_df, n_folds, model_name
        )
    
    # Save all models
    joblib.dump(group_models, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, 'group_models.pkl'))
    return group_models

if __name__ == "__main__":
    train_structural_group_models(model_name='rff', n_folds=5)