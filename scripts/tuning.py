import optuna
import xgboost as xgb
from sklearn.metrics import f1_score, zero_one_loss
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
import seaborn as sns

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

def objective(trial, X, y, groups, df):
    # Define hyperparameters to optimize
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',  # Use built-in AUC metric for training
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }
    
    # Add specific parameters for dart booster
    if param['booster'] == 'dart':
        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        param['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 0.5, log=True)
        param['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 0.5, log=True)
    
    # Initialize cross-validation
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    # Perform cross-validation
    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_valid = df.iloc[val_idx].reset_index(drop=True)
        
        # Calculate scale_pos_weight based on class imbalance in validation set
        n_neg = (y_val == 0).sum()
        n_pos = (y_val == 1).sum()
        scale_pos_weight = n_neg / n_pos
        param['scale_pos_weight'] = scale_pos_weight
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        model = xgb.train(
            param,
            dtrain,
            num_boost_round=param['n_estimators'],
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Make predictions
        y_pred_proba = model.predict(dval)
        
        # Add prediction probabilities to validation dataframe
        df_valid['prediction_probability'] = y_pred_proba
        
        # Get true roots for each sentence
        true_roots = df_valid[df_valid['is_root'] == 1][['sentence', 'language', 'node_number']]
        true_roots = true_roots.rename(columns={'node_number': 'true_root'})
        
        # Get predicted roots (nodes with max probability) for each sentence
        pred_roots = df_valid.loc[df_valid.groupby(by=['sentence', 'language'])['prediction_probability'].idxmax()]
        pred_roots = pred_roots[['sentence', 'language', 'node_number']]
        pred_roots = pred_roots.rename(columns={'node_number': 'predicted_root'})
        
        # Merge true and predicted roots
        comparison = pd.merge(true_roots, pred_roots, on=['sentence', 'language'], how='inner')
        
        # Calculate zero_one_loss
        score = 1 - zero_one_loss(comparison['true_root'], comparison['predicted_root'])
        cv_scores.append(score)
    
    # Return mean CV score
    return np.mean(cv_scores)

def main():
    # Load data
    df = pd.read_csv('../data/train_data_new.csv')
    
    # Prepare groups (using sentence as group)
    groups = df['sentence']
    
    # Prepare features and target
    X = df.drop(config.TRAIN_DROP_COLS, axis=1)
    y = df['is_root']
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Run optimization
    study.optimize(lambda trial: objective(trial, X, y, groups, df), n_trials=100)
    
    # Print best parameters and score
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train final model with best parameters
    best_params = trial.params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    best_params['random_state'] = 42
    
    # Calculate final scale_pos_weight from entire dataset
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    best_params['scale_pos_weight'] = n_neg / n_pos
    
    # Create DMatrix for final training
    dtrain = xgb.DMatrix(X, label=y)
    
    # Train final model
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=best_params['n_estimators']
    )
    
    # Get feature importance
    importance_dict = final_model.get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance)), feature_importance['importance'])
    plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Print top 10 most important features
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()


# Best XGBoosttrial with F1 Score with metric auc:
#   Value:  0.35252649284819604
#   Params: 
#     booster: gbtree
#     lambda: 1.1199566483339617e-07
#     alpha: 6.101403004331885e-07
#     learning_rate: 0.14721823854055904
#     n_estimators: 961
#     max_depth: 9
#     min_child_weight: 5
#     gamma: 0.0002965398395149084
#     subsample: 0.9820117486294455
#     colsample_bytree: 0.6166351578002189


# Best trial with F1 Score with metric logloss:
#   Value:  0.3522538423831739
#   Params: 
#     booster: dart
#     lambda: 0.00044774351548967705
#     alpha: 0.000694912939521992
#     learning_rate: 0.13821190677129985
#     n_estimators: 163
#     max_depth: 9
#     min_child_weight: 6
#     gamma: 1.6177472372323167e-05
#     subsample: 0.9519836350509681
#     colsample_bytree: 0.9424162725030374
#     sample_type: weighted
#     normalize_type: tree
#     rate_drop: 0.001911472383359846
#     skip_drop: 0.0010850466610653905


# Best trial:
#   Value:  0.3742857142857143
#   Params: 
#     booster: gbtree
#     lambda: 0.01646184064610464
#     alpha: 8.925449429850201e-07
#     learning_rate: 0.04766323397168665
#     n_estimators: 726
#     max_depth: 8
#     min_child_weight: 5
#     gamma: 1.2053850658756302e-08
#     subsample: 0.9816077266525208
#     colsample_bytree: 0.8483455947091667