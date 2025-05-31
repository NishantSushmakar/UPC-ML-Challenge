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


def zero_one_loss(y_true, y_pred):
    """
    Computes the 0-1 loss.

    Parameters:
    - y_true: list or array of true labels
    - y_pred: list or array of predicted labels

    Returns:
    - loss: float, the average 0-1 loss
    """
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of true and predicted labels must match.")

    incorrect = sum(yt != yp for yt, yp in zip(y_true, y_pred))
    loss = incorrect / len(y_true)
    return loss


def create_groupkfolds(df, n_folds, group_col):
    """
    Modified version that:
    1. First expands the dataset using the feature pipeline
    2. Then performs stratified group k-fold splitting
    3. Returns both expanded and original DataFrames with fold assignments
    """
    # First create the expanded dataset
    feature_pipeline = Pipeline(steps=[
        ("Language Features", LanguageFeature()),
        ("Graph Features", GraphFeatures()),
        ("Node Features", NodeFeatures()),
        ("Dataset Creation", FormatDataFrame()),
        ("Language One Hot Encoding", LanguageOHE(
            enc_lan="temp_lan_encoder.pkl",
            enc_lan_family="temp_lan_family_encoder.pkl"))
    ])
    
    # Make a copy of the original DataFrame
    original_df = df.copy()
    original_df['original_index'] = original_df.index
    
    # Create expanded DataFrame
    expanded_df = feature_pipeline.fit_transform(df)


    # Now perform stratified group k-fold on the expanded data
    expanded_df['kfold'] = -1
    y = expanded_df.is_root
    groups = expanded_df[group_col]

    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    # gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for f, (t_, v_) in enumerate(gkf.split(X=expanded_df, y=y, groups=groups)):
        expanded_df.loc[v_, 'kfold'] = f
    
    # Propagate fold assignments back to original DataFrame
    # Get unique (sentence, language) pairs for each fold
    fold_assignments = expanded_df[['sentence', 'language', 'kfold']].drop_duplicates()
    original_df = original_df.merge(fold_assignments, on=['sentence', 'language'], how='left')

    return original_df, expanded_df


def run_folds(original_df, expanded_df, fold, model):
    """
    Modified to work with both original and expanded DataFrames
    """
    # Get the original train/valid splits
    orig_train = original_df[original_df.kfold != fold].reset_index(drop=True)
    orig_valid = original_df[original_df.kfold == fold].reset_index(drop=True)
    
    # Get the expanded train/valid splits
    train_data = expanded_df[expanded_df.kfold != fold].reset_index(drop=True)
    valid_data = expanded_df[expanded_df.kfold == fold].reset_index(drop=True)

    if not os.path.exists(f'/Users/marwasulaiman/Documents/BDMA/UPC - 2nd semester/ML/Project/ml-project-2024-2025/UPC-ML-Challenge-RF/resources/{model}'):
        os.makedirs(f'/Users/marwasulaiman/Documents/BDMA/UPC - 2nd semester/ML/Project/ml-project-2024-2025/UPC-ML-Challenge-RF/resources/{model}')

    cols_to_use = [ 
    "eccentricity",
    "closeness_cent",
    "subgraph_cent",
    "betweeness_cent",
    "page_cent",
    "number_of_nodes",
    "num_leaf_neighbors",
    "is_leaf",
    "eigen_cent",
    "degree",
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
    x_train_data = train_data[cols_to_use]
    y_train_data = train_data.is_root.values

    # x_valid_data = valid_data.drop(columns=config.TRAIN_DROP_COLS)
    x_valid_data = valid_data[cols_to_use]
    y_valid_data = valid_data.is_root.values

    
    clf = models[model]

    # Apply oversampling only to training data
    # ros = RandomOverSampler(random_state=42)
    # x_train_resampled, y_train_resampled = ros.fit_resample(x_train_data, y_train_data)

    x_train_data = train_data[cols_to_use + ['sentence', 'language']]

    if model == "rff":
        clf.fit(x_train_data, y_train_data)

    x_train_data = x_train_data.drop(['sentence', 'language'], axis=1)

    if model == "rf":
        clf.fit(x_train_data,y_train_data)

        oob_error = clf.oob_score_
        print(f"OOB Error: {oob_error}")


    joblib.dump(clf,os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,f'{model}/{model}_{fold}.pkl'))

    y_train_pred = clf.predict(x_train_data)
    y_valid_pred = clf.predict(x_valid_data)

    y_train_proba = clf.predict_proba(x_train_data)[:,1]
    y_valid_proba = clf.predict_proba(x_valid_data)[:,1]


    ### Predicting the roots for the classes
    train_data['prediction_probability'] = y_train_proba
    valid_data['prediction_probability'] = y_valid_proba

    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    
    train_max_rows = train_data.loc[train_data.groupby(by = ['sentence','language'])['prediction_probability'].idxmax()]
    valid_max_rows = valid_data.loc[valid_data.groupby(by = ['sentence','language'])['prediction_probability'].idxmax()]

    # After getting predictions, merge with original DataFrames
    train_result = train_max_rows[['sentence','language','node_number']]
    train_result = train_result.rename(columns={'node_number':'predicted_root'})
    valid_result = valid_max_rows[['sentence','language','node_number']]
    valid_result = valid_result.rename(columns={'node_number':'predicted_root'})

    # Merge with original DataFrames instead of expanded ones
    df_train_final = pd.merge(orig_train, train_result, on=['sentence','language'], how='inner')
    df_valid_final = pd.merge(orig_valid, valid_result, on=['sentence','language'], how='inner')

    # Calculate metrics using the original-style DataFrames
    train_zero_one_loss = zero_one_loss(df_train_final['root'], df_train_final['predicted_root'])    
    valid_zero_one_loss = zero_one_loss(df_valid_final['root'], df_valid_final['predicted_root'])


    train_precision = precision_score(y_train_data, y_train_pred, average='macro')  
    train_recall = recall_score(y_train_data, y_train_pred, average='macro')
    train_f1_score = f1_score(y_train_data, y_train_pred, average='macro')
    train_roc_auc_score = roc_auc_score(y_train_data,y_train_proba)
    train_log_loss = log_loss(y_train_data,y_train_proba)
    

    valid_precision = precision_score(y_valid_data, y_valid_pred, average='macro')  
    valid_recall = recall_score(y_valid_data, y_valid_pred, average='macro')
    valid_f1_score = f1_score(y_valid_data, y_valid_pred, average='macro')
    valid_roc_auc_score = roc_auc_score(y_valid_data,y_valid_proba)
    valid_log_loss = log_loss(y_valid_data,y_valid_proba)
    
    

    print("*"*50,f"Fold {fold}","*"*50)
    print(f'Train F1 Score:{train_f1_score}, Train Precision:{train_precision}, Train Recall: {train_recall},\
          Train ROC AUC Score:{train_roc_auc_score},Train Zero One Loss:{train_zero_one_loss},Train Log Loss :{train_log_loss}')
    print(f'Validation F1 Score:{valid_f1_score}, Validation Precision:{valid_precision}, Validation Recall: {valid_recall},\
          Validation ROC AUC Score: {valid_roc_auc_score}, Valid Zero One Loss:{valid_zero_one_loss},Valid Log Loss: {valid_log_loss}')
    print("*"*100)
    
    return train_precision, train_recall, train_f1_score, train_roc_auc_score,train_zero_one_loss,train_log_loss,\
            valid_precision, valid_recall, valid_f1_score, valid_roc_auc_score,valid_zero_one_loss,valid_log_loss



class OutputRedirector:
    def __init__(self, filename, mode='w'):
        self.terminal = sys.stdout
        self.log_file = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()

def train_model(model_name, n_folds, log_file=None):
    # Create log file with timestamp if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"/Users/marwasulaiman/Documents/BDMA/UPC - 2nd semester/ML/Project/ml-project-2024-2025/UPC-ML-Challenge-RF/results/model_training_{model_name}_{timestamp}.log"
    
    # Create output redirector
    redirector = OutputRedirector(log_file)
    sys.stdout = redirector
    
    try:
        print(f"Training model: {model_name} with {n_folds} folds")
        print(f"Log file: {log_file}")
        print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("-" * 80)

        print("Loading the data")
        df = pd.read_csv(config.TRAINING_DATA_PATH)
        # df = df_orig[df_orig['language'] != "Japanese"].copy()

        original_df, expanded_df = create_groupkfolds(df, n_folds, 'sentence')

        # df = create_groupkfolds(df, n_folds, 'sentence')
        print("Loaded and Folds created successfully!!!")
        
        train_p = []
        train_r = []
        train_f1 = []
        train_auc = []
        train_zol = []
        train_lloss = []
        val_p = []
        val_r = []
        val_f1 = []
        val_auc = []
        val_zol = []
        val_lloss = []
        
        for i in range(n_folds):
            tp, tr, tf1, tauc, tzol, tlloss, vp, vr, vf1, vauc, vzol, vlloss = run_folds(original_df, expanded_df, i, model_name)
            
            train_p.append(tp)
            train_r.append(tr)
            train_f1.append(tf1)
            train_auc.append(tauc)
            train_zol.append(tzol)
            train_lloss.append(tlloss)
            val_p.append(vp)
            val_r.append(vr)
            val_f1.append(vf1)
            val_auc.append(vauc)
            val_zol.append(vzol)
            val_lloss.append(vlloss)
        
        print("*"*15,"Final Summary","*"*15)

        print(f"average training precision:{np.array(train_p).mean()}",f"std training precision:{np.array(train_p).std()}")
        print(f"average training recall:{np.array(train_r).mean()}",f"std training recall:{np.array(train_r).std()}")
        print(f"average training f1:{np.array(train_f1).mean()}",f"std training f1:{np.array(train_f1).std()}")
        print(f"average training auc:{np.array(train_auc).mean()}",f"std training auc:{np.array(train_auc).std()}")
        print(f"average training zero one loss:{np.array(train_zol).mean()}",f"std training zero one loss:{np.array(train_zol).std()}")
        print(f"average training log loss:{np.array(train_lloss).mean()}",f"std training log loss:{np.array(train_lloss).std()}")

        print(f"average validation precision:{np.array(val_p).mean()}",f"std validation precision:{np.array(val_p).std()}")
        print(f"average validation recall:{np.array(val_r).mean()}",f"std validation recall:{np.array(val_r).std()}")
        print(f"average validation f1:{np.array(val_f1).mean()}",f"std validation f1:{np.array(val_f1).std()}")
        print(f"average validation auc:{np.array(val_auc).mean()}",f"std validation f1:{np.array(val_auc).std()}")
        print(f"average validation zero one loss:{np.array(val_zol).mean()}",f"std validation zero one loss:{np.array(val_zol).std()}")
        print(f"average validation log loss:{np.array(val_lloss).mean()}",f"std validation log loss:{np.array(val_lloss).std()}")
        
        print("-" * 80)
        print("Training completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    finally:
        # Reset stdout and close log file
        sys.stdout = sys.__stdout__
        redirector.close()
        print(f"Log file saved to: {os.path.abspath(log_file)}")


if __name__ == "__main__":
    # rf for random forest, rff for sentence-aware random forest
    train_model('rff', 5) 