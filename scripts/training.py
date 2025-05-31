import pandas as pd 
import config
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,log_loss
import numpy as np
from feature_creation import *
from model_dispatcher import models
from sklearn.preprocessing import MinMaxScaler
import sys
import os
from datetime import datetime
import joblib
import networkx as nx
from typing import Dict
from gcn_model import prepare_graph_data, train_gcn_model, GCNModel
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline



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
    
    df['kfold'] = -1
    y = df.is_root
    groups = df[group_col]
    
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for f, (t_, v_) in enumerate(sgkf.split(X=df, y=y, groups=groups)):
        df.loc[v_, 'kfold'] = f
         
    return df


def run_folds_iso(df,fold):

    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    df_train['edgelist'] = df_train['edgelist'].apply(ast.literal_eval)
    df_valid['edgelist'] = df_valid['edgelist'].apply(ast.literal_eval)

    
    df_train['graph'] = df_train['edgelist'].apply(lambda edges: nx.from_edgelist(edges))
    df_valid['graph'] = df_valid['edgelist'].apply(lambda edges: nx.from_edgelist(edges))

    df_valid['predicted_iso_root'] = -1
    
    for i,valid_row in df_valid.iterrows():

        iso_count_dict = {}
        for j, train_row in df_train.iterrows():

            G1  = train_row['graph']
            G2 = valid_row['graph']
            
            GM = nx.isomorphism.GraphMatcher(G1, G2)

            if GM.is_isomorphic():
                
                root = GM.mapping[train_row['root']] 

                if root in iso_count_dict.keys():
                    iso_count_dict[root] += 1
                else:
                    iso_count_dict[root] = 1

        pred_root = -1

        if len(iso_count_dict) > 0:

 
            count = 0

            for key,value in iso_count_dict.items():

                if value > count : 
                    pred_root = key
                    count = value 


        df_valid.loc[i,'predicted_iso_root'] = pred_root

    df_valid = df_valid.drop(columns=['graph'])
    
    return df_valid
          



def run_folds(df_train, df_valid, fold, model):

    # Create model directory if it doesn't exist
    model_dir = os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    x_train_data = df_train.drop(columns=config.TRAIN_DROP_COLS)
    y_train_data = df_train.is_root.values
    
    x_valid_data = df_valid.drop(columns=config.TRAIN_DROP_COLS)
    y_valid_data = df_valid.is_root.values

    # Compute class weights for XGBoost
    if model == 'xgb':
        neg_count = np.sum(y_train_data == 0)
        pos_count = np.sum(y_train_data == 1)
        scale_pos_weight = neg_count / pos_count
        print(f"Fold {fold} - Scale pos weight: {scale_pos_weight:.2f}")
        models[model].set_params(scale_pos_weight=scale_pos_weight)
    
    clf = models[model]

    if model in ['mnb','lr']:
        scaler = MinMaxScaler()
        x_train_data = scaler.fit_transform(x_train_data)
        x_valid_data = scaler.transform(x_valid_data)
        joblib.dump(scaler,os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,f'{model}/scaler_{model}_{fold}.pkl'))

    # Train on original data
    clf.fit(x_train_data, y_train_data)
    joblib.dump(clf,os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,f'{model}/{model}_{fold}.pkl'))

    # Get predictions
    y_train_pred = clf.predict(x_train_data)
    y_valid_pred = clf.predict(x_valid_data)

    y_train_proba = clf.predict_proba(x_train_data)[:,1]
    y_valid_proba = clf.predict_proba(x_valid_data)[:,1]

    ### Predicting the roots for the classes
    df_train['prediction_probability'] = y_train_proba
    df_valid['prediction_probability'] = y_valid_proba

    # Get max probability predictions for each sentence
    train_max_rows = df_train.loc[df_train.groupby(by=['sentence', 'language'])['prediction_probability'].idxmax()]
    train_result = train_max_rows[['sentence', 'language', 'node_number']]
    train_result = train_result.rename(columns={'node_number': 'predicted_root'})

    valid_max_rows = df_valid.loc[df_valid.groupby(by=['sentence', 'language'])['prediction_probability'].idxmax()]
    valid_result = valid_max_rows[['sentence', 'language', 'node_number']]
    valid_result = valid_result.rename(columns={'node_number': 'predicted_root'})

    # Get actual roots for each sentence
    train_actual_roots = df_train[df_train.is_root == 1][['sentence', 'language', 'node_number']].reset_index(drop=True)
    train_actual_roots = train_actual_roots.rename(columns={'node_number': 'actual_root'})
    
    valid_actual_roots = df_valid[df_valid.is_root == 1][['sentence', 'language', 'node_number']].reset_index(drop=True)
    valid_actual_roots = valid_actual_roots.rename(columns={'node_number': 'actual_root'})

    # Merge predictions with actual roots
    train_comparison = pd.merge(train_result, train_actual_roots, on=['sentence', 'language'], how='inner')
    valid_comparison = pd.merge(valid_result, valid_actual_roots, on=['sentence', 'language'], how='inner')

    # Calculate zero one loss
    train_zero_one_loss = zero_one_loss(train_comparison['actual_root'], train_comparison['predicted_root'])
    valid_zero_one_loss = zero_one_loss(valid_comparison['actual_root'], valid_comparison['predicted_root'])

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
        log_file = f"../results/model_training_{model_name}_{timestamp}.log"
    
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
        print("Loaded the data successfully!!!")

        # Create feature pipeline once
        print("\nCreating feature pipeline...")
        feature_pipeline = Pipeline(steps=[
            ("Language Features", LanguageFeature()),
            ("Graph Features", GraphFeatures()),
            ("Node Features", NodeFeatures()),
            ("Dataset Creation", FormatDataFrame()),
            ("Language One Hot Encoding", LanguageOHE(
                enc_lan=f"{model_name}/lan_encoder_{model_name}_stratified.pkl",
                enc_lan_family=f"{model_name}/lan_family_encoder_{model_name}_stratified.pkl"
            ))
        ])

        # Fit the pipeline on all data to ensure consistent feature creation
        print("Fitting feature pipeline on all data...")
        df = feature_pipeline.fit_transform(df)
        df = create_groupkfolds(df, n_folds, 'sentence')
        print("Feature pipeline created and fitted successfully!")
        
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
            print(f"\nProcessing fold {i}")
            df_train = df[df.kfold != i].reset_index(drop=True)
            df_valid = df[df.kfold == i].reset_index(drop=True)
            
            tp, tr, tf1, tauc, tzol, tlloss, vp, vr, vf1, vauc, vzol, vlloss = run_folds(
                df_train, df_valid, i, model_name
            )
            
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





def train_model_per_language(model_name, n_folds, log_file=None):
    """
    Train model with a language-first approach - first select language then run folds.
    This is an alternative to train_model that processes languages first, then folds.
    
    Args:
        model_name: Name of the model to train
        n_folds: Number of folds for cross-validation
        log_file: Optional log file path
    """
    # Create log file with timestamp if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"../results/model_training_{model_name}_per_language_{timestamp}.log"
    
    # Create output redirector
    redirector = OutputRedirector(log_file)
    sys.stdout = redirector
    
    try:
        print(f"Training model: {model_name} with {n_folds} folds (Language-first approach)")
        print(f"Log file: {log_file}")
        print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("-" * 80)

        print("Loading the data")
        df = pd.read_csv(config.TRAINING_DATA_PATH)

        df = create_groupkfolds(df, n_folds, 'sentence')
        print("Loaded and Folds created successfully!!!")
        
        # Get unique languages
        languages = df['language'].unique()
        print(f"Found {len(languages)} languages: {languages}")
        
        # Dictionary to store results for each language
        language_results = {}
        
        # Process each language
        for lang in languages:
            print(f"\n{'='*20} Processing Language: {lang} {'='*20}")
            
            # Get data for current language
            df_lang = df[df.language == lang].reset_index(drop=True)
            
            if df_lang.empty:
                print(f"Skipping {lang} - no data available")
                continue
                
            # Store fold results for this language
            fold_results = []
            
            # Process each fold for current language
            for fold in range(n_folds):
                print(f"\nProcessing fold {fold} for {lang}")
                
                # Split data for this fold
                df_train = df_lang[df_lang.kfold != fold].reset_index(drop=True)
                df_valid = df_lang[df_lang.kfold == fold].reset_index(drop=True)
                
                if df_train.empty or df_valid.empty:
                    print(f"Skipping fold {fold} for {lang} - insufficient data")
                    continue
                
                # Create feature pipeline
                feature_pipeline = Pipeline(steps=[
                    ("Language Features", LanguageFeature()),
                    ("Graph Features", GraphFeatures()),
                    ("Node Features", NodeFeatures()),
                    ("Dataset Creation", FormatDataFrame())
                ])
                
                # Process data
                train_data = feature_pipeline.fit_transform(df_train)
                valid_data = feature_pipeline.transform(df_valid)
                
                x_train = train_data.drop(columns=config.TRAIN_DROP_COLS)
                y_train = train_data.is_root.values
                x_valid = valid_data.drop(columns=config.TRAIN_DROP_COLS)
                y_valid = valid_data.is_root.values
                
                # Train model
                clf = models[model_name]
                
                # Apply scaling if needed
                if model_name in ['mnb', 'lr']:
                    scaler = MinMaxScaler()
                    x_train = scaler.fit_transform(x_train)
                    x_valid = scaler.transform(x_valid)
                    joblib.dump(scaler, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, 
                                f'{model_name}/scaler_{lang}_{model_name}_{fold}.pkl'))
                
                # Train and save model
                clf.fit(x_train, y_train)
                joblib.dump(clf, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, 
                            f'{model_name}/model_{lang}_{model_name}_{fold}.pkl'))
                
                # Get predictions
                y_train_pred = clf.predict(x_train)
                y_valid_pred = clf.predict(x_valid)
                y_train_proba = clf.predict_proba(x_train)[:, 1]
                y_valid_proba = clf.predict_proba(x_valid)[:, 1]

                # Add prediction probabilities to dataframes
                train_data['prediction_probability'] = y_train_proba
                valid_data['prediction_probability'] = y_valid_proba

                # Get max probability predictions for each sentence
                train_max_rows = train_data.loc[train_data.groupby(by=['sentence', 'language'])['prediction_probability'].idxmax()]
                train_result = train_max_rows[['sentence', 'language', 'node_number', 'prediction_probability']]
                train_result = train_result.rename(columns={'node_number': 'predicted_root'})

                valid_max_rows = valid_data.loc[valid_data.groupby(by=['sentence', 'language'])['prediction_probability'].idxmax()]
                valid_result = valid_max_rows[['sentence', 'language', 'node_number', 'prediction_probability']]
                valid_result = valid_result.rename(columns={'node_number': 'predicted_root'})

                # Merge predictions back to original dataframes
                df_train = pd.merge(df_train, train_result, on=['sentence', 'language'], how='inner')
                df_valid = pd.merge(df_valid, valid_result, on=['sentence', 'language'], how='inner')

                # Calculate metrics
                train_precision = precision_score(y_train, y_train_pred, average='macro')
                train_recall = recall_score(y_train, y_train_pred, average='macro')
                train_f1_score = f1_score(y_train, y_train_pred, average='macro')
                train_roc_auc_score = roc_auc_score(y_train, y_train_proba)
                train_log_loss = log_loss(y_train, y_train_proba)
                train_zero_one_loss = zero_one_loss(df_train['root'], df_train['predicted_root'])

                valid_precision = precision_score(y_valid, y_valid_pred, average='macro')
                valid_recall = recall_score(y_valid, y_valid_pred, average='macro')
                valid_f1_score = f1_score(y_valid, y_valid_pred, average='macro')
                valid_roc_auc_score = roc_auc_score(y_valid, y_valid_proba)
                valid_log_loss = log_loss(y_valid, y_valid_proba)
                valid_zero_one_loss = zero_one_loss(df_valid['root'], df_valid['predicted_root'])
                
                # Store fold results
                fold_results.append({
                    'train_metrics': {
                        'precision': train_precision,
                        'recall': train_recall,
                        'f1_score': train_f1_score,
                        'roc_auc': train_roc_auc_score,
                        'log_loss': train_log_loss,
                        'zero_one_loss': train_zero_one_loss
                    },
                    'valid_metrics': {
                        'precision': valid_precision,
                        'recall': valid_recall,
                        'f1_score': valid_f1_score,
                        'roc_auc': valid_roc_auc_score,
                        'log_loss': valid_log_loss,
                        'zero_one_loss': valid_zero_one_loss
                    }
                })
                
                # Print fold results
                print(f"\nFold {fold} results for {lang}:")
                print(f"Train - F1: {train_f1_score:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
                print(f"Train - Zero One Loss: {train_zero_one_loss:.4f}, Log Loss: {train_log_loss:.4f}")
                print(f"Valid - F1: {valid_f1_score:.4f}, Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}")
                print(f"Valid - Zero One Loss: {valid_zero_one_loss:.4f}, Log Loss: {valid_log_loss:.4f}")
            
            # Calculate average metrics across folds for this language
            if fold_results:
                avg_train_metrics = {
                    metric: np.mean([fold['train_metrics'][metric] for fold in fold_results])
                    for metric in ['precision', 'recall', 'f1_score', 'roc_auc', 'log_loss', 'zero_one_loss']
                }
                
                avg_valid_metrics = {
                    metric: np.mean([fold['valid_metrics'][metric] for fold in fold_results])
                    for metric in ['precision', 'recall', 'f1_score', 'roc_auc', 'log_loss', 'zero_one_loss']
                }
                
                # Store language results
                language_results[lang] = {
                    'fold_results': fold_results,
                    'avg_train_metrics': avg_train_metrics,
                    'avg_valid_metrics': avg_valid_metrics
                }
                
                # Print language summary
                print(f"\n{'-'*20} Summary for {lang} {'-'*20}")
                print(f"Train - F1: {avg_train_metrics['f1_score']:.4f}, Precision: {avg_train_metrics['precision']:.4f}, Recall: {avg_train_metrics['recall']:.4f}")
                print(f"Train - Zero One Loss: {avg_train_metrics['zero_one_loss']:.4f}, Log Loss: {avg_train_metrics['log_loss']:.4f}")
                print(f"Valid - F1: {avg_valid_metrics['f1_score']:.4f}, Precision: {avg_valid_metrics['precision']:.4f}, Recall: {avg_valid_metrics['recall']:.4f}")
                print(f"Valid - Zero One Loss: {avg_valid_metrics['zero_one_loss']:.4f}, Log Loss: {avg_valid_metrics['log_loss']:.4f}")
        
        # Calculate final averages across all languages
        if language_results:
            final_train_metrics = {
                metric: np.mean([results['avg_train_metrics'][metric] for results in language_results.values()])
                for metric in ['precision', 'recall', 'f1_score', 'roc_auc', 'log_loss', 'zero_one_loss']
            }
            
            final_valid_metrics = {
                metric: np.mean([results['avg_valid_metrics'][metric] for results in language_results.values()])
                for metric in ['precision', 'recall', 'f1_score', 'roc_auc', 'log_loss', 'zero_one_loss']
            }
            
            print("\n" + "*"*15 + "Final Summary" + "*"*15)
            print("\nAverage metrics across all languages:")
            print(f"Train - F1: {final_train_metrics['f1_score']:.4f}, Precision: {final_train_metrics['precision']:.4f}, Recall: {final_train_metrics['recall']:.4f}")
            print(f"Train - Zero One Loss: {final_train_metrics['zero_one_loss']:.4f}, Log Loss: {final_train_metrics['log_loss']:.4f}")
            print(f"Valid - F1: {final_valid_metrics['f1_score']:.4f}, Precision: {final_valid_metrics['precision']:.4f}, Recall: {final_valid_metrics['recall']:.4f}")
            print(f"Valid - Zero One Loss: {final_valid_metrics['zero_one_loss']:.4f}, Log Loss: {final_valid_metrics['log_loss']:.4f}")
        
        print("-" * 80)
        print("Training completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    finally:
        # Reset stdout and close log file
        sys.stdout = sys.__stdout__
        redirector.close()
        print(f"Log file saved to: {os.path.abspath(log_file)}")
        
    return language_results


def create_balanced_sum_folds(df, n_folds, group_col, sum_col='n'):
    """
    Creates folds while trying to maintain equal sum of a specified column across folds.
    
    Parameters:
    - df: pandas DataFrame containing the data
    - n_folds: number of folds to create
    - group_col: column name to use for grouping (e.g., 'sentence')
    - sum_col: column name whose sum should be balanced across folds (default: 'n')
    
    Returns:
    - DataFrame with added 'kfold' column
    """
    df['kfold'] = -1
    
    # Get unique groups and their corresponding sums
    group_sums = df.groupby(group_col)[sum_col].sum().reset_index()
    group_sums = group_sums.sort_values(by=sum_col, ascending=False)
    
    # Initialize fold sums
    fold_sums = [0] * n_folds
    
    # Assign groups to folds
    for _, row in group_sums.iterrows():
        # Find the fold with minimum current sum
        min_fold = np.argmin(fold_sums)
        
        # Assign all rows from this group to the selected fold
        group_mask = df[group_col] == row[group_col]
        df.loc[group_mask, 'kfold'] = min_fold
        
        # Update the fold sum
        fold_sums[min_fold] += row[sum_col]
    
    return df


def run_folds_split_models(df, fold, model_name, japanese_model_name):
    """
    Run folds with separate models for Japanese and other languages, but combine predictions
    for final evaluation. Saves both models separately.
    
    Parameters:
    - df: DataFrame containing the data
    - fold: Current fold number
    - model_name: Model to use for non-Japanese languages
    - japanese_model_name: Model to use for Japanese language
    
    Returns:
    - Combined metrics for all languages
    """
    # Split data for current fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # Create directories for both models if they don't exist
    for model in [model_name, japanese_model_name]:
        if not os.path.exists(f'../resources/{model}'):
            os.makedirs(f'../resources/{model}')
    
    # Create feature pipeline
    feature_pipeline = Pipeline(steps=[
        ("Language Features", LanguageFeature()),
        ("Graph Features", GraphFeatures()),
        ("Node Features", NodeFeatures()),
        ("Dataset Creation", FormatDataFrame()),
        ("Language One Hot Encoding", LanguageOHE(
            enc_lan=f"{model_name}/lan_encoder_{model_name}_{fold}.pkl",
            enc_lan_family=f"{model_name}/lan_family_encoder_{model_name}_{fold}.pkl"
        ))
    ])
    
    # Process all data through feature pipeline
    train_data = feature_pipeline.fit_transform(df_train)
    valid_data = feature_pipeline.transform(df_valid)
    
    # Split data for Japanese and non-Japanese
    jp_train = train_data[train_data.language == 'Japanese'].reset_index(drop=True)
    jp_valid = valid_data[valid_data.language == 'Japanese'].reset_index(drop=True)
    other_train = train_data[train_data.language != 'Japanese'].reset_index(drop=True)
    other_valid = valid_data[valid_data.language != 'Japanese'].reset_index(drop=True)
    
    # Function to train model and get predictions
    def train_and_predict(train_data, valid_data, model_name, is_japanese=False):
        if train_data.empty or valid_data.empty:
            return None, None, None, None
        
        x_train = train_data.drop(columns=config.TRAIN_DROP_COLS)
        y_train = train_data.is_root.values
        x_valid = valid_data.drop(columns=config.TRAIN_DROP_COLS)
        y_valid = valid_data.is_root.values
        
        clf = models[model_name]
        
        # Save scaler with appropriate naming
        if model_name in ['mnb', 'lr']:
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train)
            x_valid = scaler.transform(x_valid)
            scaler_name = f"{model_name}/scaler_{model_name}_{'jp' if is_japanese else 'other'}_{fold}.pkl"
            joblib.dump(scaler, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, scaler_name))
        
        # Train and save model with appropriate naming
        clf.fit(x_train, y_train)
        model_name_save = f"{model_name}/{model_name}_{'jp' if is_japanese else 'other'}_{fold}.pkl"
        joblib.dump(clf, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, model_name_save))
        
        y_train_pred = clf.predict(x_train)
        y_valid_pred = clf.predict(x_valid)
        y_train_proba = clf.predict_proba(x_train)[:, 1]
        y_valid_proba = clf.predict_proba(x_valid)[:, 1]
        
        # Add predictions to the original dataframes to maintain alignment
        train_data['prediction'] = y_train_pred
        train_data['prediction_probability'] = y_train_proba
        valid_data['prediction'] = y_valid_pred
        valid_data['prediction_probability'] = y_valid_proba
        
        return train_data, valid_data
    
    # Train models and get predictions with data
    jp_train_data, jp_valid_data = train_and_predict(jp_train, jp_valid, japanese_model_name, is_japanese=True)
    other_train_data, other_valid_data = train_and_predict(other_train, other_valid, model_name, is_japanese=False)
    
    # Combine data while preserving the original order
    def combine_data(data1, data2):
        if data1 is None:
            return data2
        if data2 is None:
            return data1
        
        # Combine while preserving the original order
        combined = pd.concat([data1, data2])
        # Sort by sentence and language to ensure consistent ordering
        combined = combined.sort_values(['sentence', 'language'])
        return combined
    
    # Combine train and validation data
    train_data = combine_data(jp_train_data, other_train_data)
    valid_data = combine_data(jp_valid_data, other_valid_data)
    
    # Get max probability predictions for each sentence
    train_max_rows = train_data.loc[train_data.groupby(by=['sentence', 'language'])['prediction_probability'].idxmax()]
    train_result = train_max_rows[['sentence', 'language', 'node_number']]
    train_result = train_result.rename(columns={'node_number': 'predicted_root'})
    
    valid_max_rows = valid_data.loc[valid_data.groupby(by=['sentence', 'language'])['prediction_probability'].idxmax()]
    valid_result = valid_max_rows[['sentence', 'language', 'node_number']]
    valid_result = valid_result.rename(columns={'node_number': 'predicted_root'})
    
    # Merge predictions back to original dataframes
    df_train = pd.merge(df_train, train_result, on=['sentence', 'language'], how='inner')
    df_valid = pd.merge(df_valid, valid_result, on=['sentence', 'language'], how='inner')
    
    # Calculate metrics using the original dataframes
    train_zero_one_loss = zero_one_loss(df_train['root'], df_train['predicted_root'])
    valid_zero_one_loss = zero_one_loss(df_valid['root'], df_valid['predicted_root'])
    
    # Calculate other metrics using the combined data
    train_precision = precision_score(train_data['is_root'], train_data['prediction'], average='macro')
    train_recall = recall_score(train_data['is_root'], train_data['prediction'], average='macro')
    train_f1_score = f1_score(train_data['is_root'], train_data['prediction'], average='macro')
    train_roc_auc_score = roc_auc_score(train_data['is_root'], train_data['prediction_probability'])
    train_log_loss = log_loss(train_data['is_root'], train_data['prediction_probability'])
    
    valid_precision = precision_score(valid_data['is_root'], valid_data['prediction'], average='macro')
    valid_recall = recall_score(valid_data['is_root'], valid_data['prediction'], average='macro')
    valid_f1_score = f1_score(valid_data['is_root'], valid_data['prediction'], average='macro')
    valid_roc_auc_score = roc_auc_score(valid_data['is_root'], valid_data['prediction_probability'])
    valid_log_loss = log_loss(valid_data['is_root'], valid_data['prediction_probability'])
    
    # Print results
    print("*"*50, f"Fold {fold}", "*"*50)
    print(f'Train F1 Score:{train_f1_score}, Train Precision:{train_precision}, Train Recall: {train_recall},')
    print(f'Train ROC AUC Score:{train_roc_auc_score}, Train Zero One Loss:{train_zero_one_loss}, Train Log Loss:{train_log_loss}')
    print(f'Validation F1 Score:{valid_f1_score}, Validation Precision:{valid_precision}, Validation Recall: {valid_recall},')
    print(f'Validation ROC AUC Score: {valid_roc_auc_score}, Valid Zero One Loss:{valid_zero_one_loss}, Valid Log Loss: {valid_log_loss}')
    print("*"*100)
    
    return (train_precision, train_recall, train_f1_score, train_roc_auc_score,
            train_zero_one_loss, train_log_loss, valid_precision, valid_recall,
            valid_f1_score, valid_roc_auc_score, valid_zero_one_loss, valid_log_loss)


def train_model_ensemble(model_configs, n_folds, log_file):
    """
    Train an ensemble of models using k-fold cross-validation.
    
    Parameters:
    - model_configs: List of dictionaries containing model configurations
    - n_folds: Number of folds for cross-validation
    - log_file: Name of the log file to save results
    """
    # Create log file with timestamp if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"../results/ensemble_training_{timestamp}.log"
    
    # Create output redirector
    redirector = OutputRedirector(log_file)
    sys.stdout = redirector
    
    try:
        print(f"Training ensemble model with {n_folds} folds")
        print(f"Log file: {log_file}")
        print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("-" * 80)

        print("Loading the data")
        df = pd.read_csv(config.TRAINING_DATA_PATH)
        print("Loaded the data successfully!!!")

        # Create feature pipeline once
        print("\nCreating feature pipeline...")
        feature_pipeline = Pipeline(steps=[
            ("Language Features", LanguageFeature()),
            ("Graph Features", GraphFeatures()),
            ("Node Features", NodeFeatures()),
            ("Dataset Creation", FormatDataFrame()),
            ("Language One Hot Encoding", LanguageOHE(
                enc_lan=f"ensemble/lan_encoder_ensemble_stratified.pkl",
                enc_lan_family=f"ensemble/lan_family_encoder_ensemble_stratified.pkl"
            ))
        ])

        # Fit the pipeline on all data to ensure consistent feature creation
        print("Fitting feature pipeline on all data...")
        df = feature_pipeline.fit_transform(df)
        df = create_groupkfolds(df, n_folds, 'sentence')
        print("Feature pipeline created and fitted successfully!")

        train_metrics = []
        valid_metrics = []

        for fold in range(n_folds):
            print(f"\nProcessing fold {fold}")
            df_train = df[df.kfold != fold].reset_index(drop=True)
            df_valid = df[df.kfold == fold].reset_index(drop=True)
            
            metrics = run_folds_ensemble(df_train, df_valid, fold, model_configs)
            train_metrics.append(metrics[:6])  # First 6 metrics are for training
            valid_metrics.append(metrics[6:])  # Last 6 metrics are for validation

        # Calculate and print average metrics
        avg_train_metrics = np.mean(train_metrics, axis=0)
        avg_valid_metrics = np.mean(valid_metrics, axis=0)
        std_train_metrics = np.std(train_metrics, axis=0)
        std_valid_metrics = np.std(valid_metrics, axis=0)

        print("\nAverage Metrics Across Folds:")
        print("\nTraining Metrics:")
        print(f"Precision: {avg_train_metrics[0]:.4f} ± {std_train_metrics[0]:.4f}")
        print(f"Recall: {avg_train_metrics[1]:.4f} ± {std_train_metrics[1]:.4f}")
        print(f"F1 Score: {avg_train_metrics[2]:.4f} ± {std_train_metrics[2]:.4f}")
        print(f"ROC AUC: {avg_train_metrics[3]:.4f} ± {std_train_metrics[3]:.4f}")
        print(f"Zero One Loss: {avg_train_metrics[4]:.4f} ± {std_train_metrics[4]:.4f}")
        print(f"Log Loss: {avg_train_metrics[5]:.4f} ± {std_train_metrics[5]:.4f}")

        print("\nValidation Metrics:")
        print(f"Precision: {avg_valid_metrics[0]:.4f} ± {std_valid_metrics[0]:.4f}")
        print(f"Recall: {avg_valid_metrics[1]:.4f} ± {std_valid_metrics[1]:.4f}")
        print(f"F1 Score: {avg_valid_metrics[2]:.4f} ± {std_valid_metrics[2]:.4f}")
        print(f"ROC AUC: {avg_valid_metrics[3]:.4f} ± {std_valid_metrics[3]:.4f}")
        print(f"Zero One Loss: {avg_valid_metrics[4]:.4f} ± {std_valid_metrics[4]:.4f}")
        print(f"Log Loss: {avg_valid_metrics[5]:.4f} ± {std_valid_metrics[5]:.4f}")

        # Save results to log file
        with open(log_file, 'w') as f:
            f.write("Average Metrics Across Folds:\n\n")
            f.write("Training Metrics:\n")
            f.write(f"Precision: {avg_train_metrics[0]:.4f} ± {std_train_metrics[0]:.4f}\n")
            f.write(f"Recall: {avg_train_metrics[1]:.4f} ± {std_train_metrics[1]:.4f}\n")
            f.write(f"F1 Score: {avg_train_metrics[2]:.4f} ± {std_train_metrics[2]:.4f}\n")
            f.write(f"ROC AUC: {avg_train_metrics[3]:.4f} ± {std_train_metrics[3]:.4f}\n")
            f.write(f"Zero One Loss: {avg_train_metrics[4]:.4f} ± {std_train_metrics[4]:.4f}\n")
            f.write(f"Log Loss: {avg_train_metrics[5]:.4f} ± {std_train_metrics[5]:.4f}\n\n")

            f.write("Validation Metrics:\n")
            f.write(f"Precision: {avg_valid_metrics[0]:.4f} ± {std_valid_metrics[0]:.4f}\n")
            f.write(f"Recall: {avg_valid_metrics[1]:.4f} ± {std_valid_metrics[1]:.4f}\n")
            f.write(f"F1 Score: {avg_valid_metrics[2]:.4f} ± {std_valid_metrics[2]:.4f}\n")
            f.write(f"ROC AUC: {avg_valid_metrics[3]:.4f} ± {std_valid_metrics[3]:.4f}\n")
            f.write(f"Zero One Loss: {avg_valid_metrics[4]:.4f} ± {std_valid_metrics[4]:.4f}\n")
            f.write(f"Log Loss: {avg_valid_metrics[5]:.4f} ± {std_valid_metrics[5]:.4f}\n\n")

            f.write("\nPer-fold results:\n")
            for fold in range(n_folds):
                f.write(f"\nFold {fold}:\n")
                f.write("Training Metrics:\n")
                f.write(f"  Precision: {train_metrics[fold][0]:.4f}\n")
                f.write(f"  Recall: {train_metrics[fold][1]:.4f}\n")
                f.write(f"  F1 Score: {train_metrics[fold][2]:.4f}\n")
                f.write(f"  ROC AUC: {train_metrics[fold][3]:.4f}\n")
                f.write(f"  Zero One Loss: {train_metrics[fold][4]:.4f}\n")
                f.write(f"  Log Loss: {train_metrics[fold][5]:.4f}\n")
                f.write("Validation Metrics:\n")
                f.write(f"  Precision: {valid_metrics[fold][0]:.4f}\n")
                f.write(f"  Recall: {valid_metrics[fold][1]:.4f}\n")
                f.write(f"  F1 Score: {valid_metrics[fold][2]:.4f}\n")
                f.write(f"  ROC AUC: {valid_metrics[fold][3]:.4f}\n")
                f.write(f"  Zero One Loss: {valid_metrics[fold][4]:.4f}\n")
                f.write(f"  Log Loss: {valid_metrics[fold][5]:.4f}\n")

        print("-" * 80)
        print("Training completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    finally:
        # Reset stdout and close log file
        sys.stdout = sys.__stdout__
        redirector.close()
        print(f"Log file saved to: {os.path.abspath(log_file)}")

def run_folds_ensemble(df_train, df_valid, fold, model_configs):
    """
    Train and evaluate an ensemble of models for a specific fold.
    
    Parameters:
    - df_train: Training DataFrame
    - df_valid: Validation DataFrame
    - fold: Current fold number
    - model_configs: List of dictionaries containing model configurations
        Each dict should have:
        - model_name: Name of the model
        - weight: Weight to give to this model's predictions (default: 1.0)
    """
    # Create directories for all models
    for m_config in model_configs:
        model_name = m_config['model_name']
        model_dir = os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    # Initialize DataFrames to store combined predictions
    train_combined = None
    valid_combined = None

    # Calculate total weight for normalization
    total_weight = sum(m_config.get('weight', 1.0) for m_config in model_configs)

    # Train and evaluate each model
    for m_config in model_configs:
        model_name = m_config['model_name']
        weight = m_config.get('weight', 1.0)
        
        print(f"\nProcessing model: {model_name} (weight: {weight})")
        
        # Prepare features
        x_train = df_train.drop(columns=config.TRAIN_DROP_COLS)
        y_train = df_train.is_root.values
        x_valid = df_valid.drop(columns=config.TRAIN_DROP_COLS)
        y_valid = df_valid.is_root.values

        # Train model
        clf = models[model_name]
        if model_name in ['mnb', 'lr']:
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train)
            x_valid = scaler.transform(x_valid)
            joblib.dump(scaler, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, f'{model_name}/scaler_{model_name}_{fold}.pkl'))

        clf.fit(x_train, y_train)
        joblib.dump(clf, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, f'{model_name}/{model_name}_{fold}.pkl'))

        # Get predictions and apply weight
        train_pred_proba = clf.predict_proba(x_train)[:, 1] * weight
        valid_pred_proba = clf.predict_proba(x_valid)[:, 1] * weight

        # Store predictions
        if train_combined is None:
            train_combined = df_train.copy()
            train_combined['prediction_probability'] = train_pred_proba
        else:
            train_combined['prediction_probability'] += train_pred_proba

        if valid_combined is None:
            valid_combined = df_valid.copy()
            valid_combined['prediction_probability'] = valid_pred_proba
        else:
            valid_combined['prediction_probability'] += valid_pred_proba

    # Normalize the combined probabilities by dividing by total weight
    train_combined['prediction_probability'] = train_combined['prediction_probability'] / total_weight
    valid_combined['prediction_probability'] = valid_combined['prediction_probability'] / total_weight

    # Get max probability predictions
    train_max_rows = train_combined.loc[train_combined.groupby(by=['sentence', 'language'])['prediction_probability'].idxmax()]
    train_result = train_max_rows[['sentence', 'language', 'node_number']]
    train_result = train_result.rename(columns={'node_number': 'predicted_root'})

    valid_max_rows = valid_combined.loc[valid_combined.groupby(by=['sentence', 'language'])['prediction_probability'].idxmax()]
    valid_result = valid_max_rows[['sentence', 'language', 'node_number']]
    valid_result = valid_result.rename(columns={'node_number': 'predicted_root'})

    # Get actual roots for each sentence
    train_actual_roots = df_train[df_train.is_root == 1][['sentence', 'language', 'node_number']].reset_index(drop=True)
    train_actual_roots = train_actual_roots.rename(columns={'node_number': 'actual_root'})
    
    valid_actual_roots = df_valid[df_valid.is_root == 1][['sentence', 'language', 'node_number']].reset_index(drop=True)
    valid_actual_roots = valid_actual_roots.rename(columns={'node_number': 'actual_root'})

    # Merge predictions with actual roots
    train_comparison = pd.merge(train_result, train_actual_roots, on=['sentence', 'language'], how='inner')
    valid_comparison = pd.merge(valid_result, valid_actual_roots, on=['sentence', 'language'], how='inner')

    # Calculate zero one loss
    train_zero_one_loss = zero_one_loss(train_comparison['actual_root'], train_comparison['predicted_root'])
    valid_zero_one_loss = zero_one_loss(valid_comparison['actual_root'], valid_comparison['predicted_root'])

    # Calculate other metrics using normalized probabilities
    train_precision = precision_score(train_combined['is_root'], train_combined['prediction_probability'] > 0.5, average='macro')
    train_recall = recall_score(train_combined['is_root'], train_combined['prediction_probability'] > 0.5, average='macro')
    train_f1_score = f1_score(train_combined['is_root'], train_combined['prediction_probability'] > 0.5, average='macro')
    train_roc_auc_score = roc_auc_score(train_combined['is_root'], train_combined['prediction_probability'])
    train_log_loss = log_loss(train_combined['is_root'], train_combined['prediction_probability'])

    valid_precision = precision_score(valid_combined['is_root'], valid_combined['prediction_probability'] > 0.5, average='macro')
    valid_recall = recall_score(valid_combined['is_root'], valid_combined['prediction_probability'] > 0.5, average='macro')
    valid_f1_score = f1_score(valid_combined['is_root'], valid_combined['prediction_probability'] > 0.5, average='macro')
    valid_roc_auc_score = roc_auc_score(valid_combined['is_root'], valid_combined['prediction_probability'])
    valid_log_loss = log_loss(valid_combined['is_root'], valid_combined['prediction_probability'])

    print("*"*50, f"Fold {fold}", "*"*50)
    print(f'Train F1 Score:{train_f1_score}, Train Precision:{train_precision}, Train Recall: {train_recall},')
    print(f'Train ROC AUC Score:{train_roc_auc_score}, Train Zero One Loss:{train_zero_one_loss}, Train Log Loss:{train_log_loss}')
    print(f'Validation F1 Score:{valid_f1_score}, Validation Precision:{valid_precision}, Validation Recall: {valid_recall},')
    print(f'Validation ROC AUC Score: {valid_roc_auc_score}, Valid Zero One Loss:{valid_zero_one_loss}, Valid Log Loss: {valid_log_loss}')
    print("*"*100)

    return (train_precision, train_recall, train_f1_score, train_roc_auc_score,
            train_zero_one_loss, train_log_loss, valid_precision, valid_recall,
            valid_f1_score, valid_roc_auc_score, valid_zero_one_loss, valid_log_loss)

def train_model_gcn(model_config: Dict, n_folds: int, log_file: str = None):
    """
    Train GCN model using k-fold cross-validation.
    
    Args:
        model_config: Dictionary containing model configuration
        n_folds: Number of folds for cross-validation
        log_file: Optional log file path
    """
    # Create log file with timestamp if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"../results/gcn_training_{timestamp}.log"
    
    # Create output redirector
    redirector = OutputRedirector(log_file)
    sys.stdout = redirector
    
    try:
        print(f"Training GCN model with {n_folds} folds")
        print(f"Log file: {log_file}")
        print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("-" * 80)

        print("Loading the data")
        df = pd.read_csv(config.TRAINING_DATA_PATH)
        df = create_groupkfolds(df, n_folds, 'sentence')
        print("Loaded and Folds created successfully!!!")
        
        # Store metrics for each fold
        fold_metrics = []
        
        for fold in range(n_folds):
            print(f"\nProcessing fold {fold}")
            
            # Split data
            df_train = df[df.kfold != fold].reset_index(drop=True)
            df_valid = df[df.kfold == fold].reset_index(drop=True)
            
            # Prepare graph data
            train_data = prepare_graph_data(df_train)
            valid_data = prepare_graph_data(df_valid)
            
            # Train model
            model, metrics = train_gcn_model(train_data, valid_data, model_config, fold)
            fold_metrics.append(metrics)
            
            # Print fold results
            print(f"\nFold {fold} results:")
            print(f"Best validation F1: {max(metrics['val_f1']):.4f}")
            print(f"Best validation loss: {min(metrics['val_loss']):.4f}")
        
        # Calculate and print average metrics
        avg_metrics = {
            'train_loss': np.mean([m['train_loss'][-1] for m in fold_metrics]),
            'val_loss': np.mean([m['val_loss'][-1] for m in fold_metrics]),
            'val_f1': np.mean([max(m['val_f1']) for m in fold_metrics])
        }
        
        print("\n" + "*"*15 + "Final Summary" + "*"*15)
        print(f"Average training loss: {avg_metrics['train_loss']:.4f}")
        print(f"Average validation loss: {avg_metrics['val_loss']:.4f}")
        print(f"Average best validation F1: {avg_metrics['val_f1']:.4f}")
        
        print("-" * 80)
        print("Training completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    finally:
        # Reset stdout and close log file
        sys.stdout = sys.__stdout__
        redirector.close()
        print(f"Log file saved to: {os.path.abspath(log_file)}")


if __name__ == "__main__":
    # train_model('xgb_auc',10,'../results/xgb_auc_balanced_with_imp_tree_features.log') 
    # train_model('xgb_logloss',10,'../results/xgb_logloss_balanced_with_imp_tree_features.log')
    
    train_model('lgbm_logloss',10,'../results/final_results/lgbm_logloss_adhoc_balanced_with_imp_tree_features_stratified.log')
    # train_model('lgbm_zero_one_loss',10,'../results/final_results/lgbm_zero_one_loss_balanced_with_imp_tree_features_stratified.log')
    # train_model('xgb_auc',10,'../results/final_results/xgb_auc_balanced_with_imp_tree_features_stratified.log')
    # train_model('xgb_logloss',10,'../results/final_results/xgb_logloss_balanced_with_imp_tree_features_stratified.log')
    # train_model('xgb_zero_one_loss',10,'../results/final_results/xgb_zero_one_loss_balanced_with_imp_tree_features_stratified.log')
    # train_model('mnb',10,'../results/final_results/mnb_balanced_with_imp_tree_features_stratified.log')
    # train_model('lr',10,'../results/final_results/lr_balanced_with_imp_tree_features_stratified.log')
    # train_model('lda',10,'../results/final_results/lda_balanced_with_imp_tree_features_stratified.log')
    # train_model('rf',10,'../results/final_results/rf_balanced_with_imp_tree_features_stratified.log')
    # train_model('lr',10,'../results/lr_balanced_with_imp_tree_features_heavy_regularization.log')
    # train_model('lgbm_zero_one_loss',10,'../results/lgbm_zero_one_loss_balanced_with_imp_tree_features.log')
    # train_model('xgb_zero_one_loss',10,'../results/xgb_zero_one_loss_balanced_with_imp_tree_features.log')

    # model_configs = [
    #     {'model_name': 'lgbm_logloss', 'weight': 0.7},
    #     {'model_name': 'lgbm_zero_one_loss', 'weight': 0.3}
    # ]
    # train_model_ensemble(model_configs, 10, '../results/final_results/lgbm_ensemble_logloss_zero_one_loss_stratified.log')


    # model_configs = [
    #     {'model_name': 'lgbm_logloss', 'weight': 0.5},
    #     {'model_name': 'xgb_logloss', 'weight': 0.3},
    #     {'model_name': 'lr', 'weight': 0.2}
    # ]
    # train_model_ensemble(model_configs, 10, '../results/final_results/lgbm_xgb_lr_ensemble_logloss_stratified.log')



    