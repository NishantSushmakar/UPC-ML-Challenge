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
    y = df.root
    groups = df[group_col]
    
    gkf = GroupKFold(n_splits=n_folds,shuffle=True,random_state=42)
    
    for f, (t_, v_) in enumerate(gkf.split(X=df, y=y, groups=groups)):
        df.loc[v_, 'kfold'] = f
         
    return df


def run_folds(df, fold, model):

    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    if not os.path.exists(f'../resources/{model}'):
        os.makedirs(f'../resources/{model}')
    

    feature_pipeline = Pipeline(steps=[
                ("Language Features",LanguageFeature()),
                ("Graph Features",GraphFeatures()),
                ("Node Features",NodeFeatures()),
                ("Dataset Creation",FormatDataFrame()),
                ("Language One Hot Encoding",LanguageOHE(enc_lan=f"{model}/lan_encoder_{model}_{fold}.pkl",\
                                                         enc_lan_family=f"{model}/lan_family_encoder_{model}_{fold}.pkl"))  
            ])
    

    train_data = feature_pipeline.fit_transform(df_train)
    valid_data = feature_pipeline.transform(df_valid)
    
    
    x_train_data = train_data.drop(columns=config.TRAIN_DROP_COLS)
    y_train_data = train_data.is_root.values
    

    x_valid_data = valid_data.drop(columns=config.TRAIN_DROP_COLS)
    y_valid_data = valid_data.is_root.values

    
    clf = models[model]


    if model in ['mnb','lr']:
        scaler = MinMaxScaler()
        x_train_data = scaler.fit_transform(x_train_data)
        x_valid_data = scaler.transform(x_valid_data)
        joblib.dump(scaler,os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,f'{model}/scaler_{model}_{fold}.pkl'))

    clf.fit(x_train_data,y_train_data)
    joblib.dump(clf,os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,f'{model}/{model}_{fold}.pkl'))

    y_train_pred = clf.predict(x_train_data)
    y_valid_pred = clf.predict(x_valid_data)

    y_train_proba = clf.predict_proba(x_train_data)[:,1]
    y_valid_proba = clf.predict_proba(x_valid_data)[:,1]


    ### Predicting the roots for the classes
    train_data['prediction_probability'] = y_train_proba
    valid_data['prediction_probability'] = y_valid_proba
    
    train_max_rows = train_data.loc[train_data.groupby(by = ['sentence','language'])['prediction_probability'].idxmax()]
    train_result = train_max_rows[['sentence','language','node_number']]
    train_result = train_result.rename(columns={'node_number':'predicted_root'})
    

    valid_max_rows = valid_data.loc[valid_data.groupby(by = ['sentence','language'])['prediction_probability'].idxmax()]
    valid_result = valid_max_rows[['sentence','language','node_number']]
    valid_result = valid_result.rename(columns={'node_number':'predicted_root'})

    df_train = pd.merge(df_train,train_result,on=['sentence','language'],how='inner')
    df_valid = pd.merge(df_valid,valid_result,on=['sentence','language'],how='inner')

    train_zero_one_loss = zero_one_loss(df_train['root'],df_train['predicted_root'])    
    valid_zero_one_loss = zero_one_loss(df_valid['root'],df_valid['predicted_root'])


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
        df = create_groupkfolds(df, n_folds, 'sentence')
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
            tp, tr, tf1, tauc, tzol, tlloss, vp, vr, vf1, vauc, vzol, vlloss = run_folds(df, i, model_name)
            
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
    # You can specify a custom log file name or let it create one automatically
    # train_model('mnb', 5, "custom_log_file.log")
    train_model('lr', 5) 