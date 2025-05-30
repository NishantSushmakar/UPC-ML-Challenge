import optuna
import lightgbm as lgb
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
        'objective': 'binary',
        'metric': 'binary_logloss',  # Use log loss metric for training
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42,
        'verbose': -1  # Disable verbose output
    }
    
    # Add specific parameters for dart booster
    if param['boosting_type'] == 'dart':
        param['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
        param['skip_drop'] = trial.suggest_float('skip_drop', 0.1, 0.5)
    
    # Add specific parameters for goss booster
    if param['boosting_type'] == 'goss':
        param['top_rate'] = trial.suggest_float('top_rate', 0.2, 0.3)
        param['other_rate'] = trial.suggest_float('other_rate', 0.1, 0.2)
    
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
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            param,
            train_data,
            num_boost_round=param['n_estimators'],
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        # Make predictions
        y_pred_proba = model.predict(X_val)
        
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
    best_params['objective'] = 'binary'
    best_params['metric'] = 'binary_logloss'
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    
    # Calculate final scale_pos_weight from entire dataset
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    best_params['scale_pos_weight'] = n_neg / n_pos
    
    # Create LightGBM dataset for final training
    train_data = lgb.Dataset(X, label=y)
    
    # Train final model
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=best_params['n_estimators']
    )
    
    # Get feature importance
    importance_dict = dict(zip(X.columns, final_model.feature_importance(importance_type='gain')))
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

# LGBM with Logloss and F1 Score:
# boosting_type: dart
#     num_leaves: 100
#     learning_rate: 0.2404304987820357
#     n_estimators: 993
#     max_depth: 8
#     min_child_samples: 98
#     subsample: 0.7758415112715576
#     colsample_bytree: 0.8449875451391139
#     reg_alpha: 8.410243613130436e-08
#     reg_lambda: 1.107002443010181e-08
#     drop_rate: 0.17665455447126532
#     skip_drop: 0.3917329494953023



# Value:  0.3778095238095238
#   Params: 
#     boosting_type: dart
#     num_leaves: 90
#     learning_rate: 0.074831705371424
#     n_estimators: 967
#     max_depth: 6
#     min_child_samples: 63
#     subsample: 0.822129055301889
#     colsample_bytree: 0.769131350148678
#     reg_alpha: 0.00013658550129263758
#     reg_lambda: 1.6283011289438004e-07
#     drop_rate: 0.14187438966796803
#     skip_drop: 0.4876667967884481