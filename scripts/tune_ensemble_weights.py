import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import config
from feature_creation import *
from model_dispatcher import models
import joblib
import os
from datetime import datetime
from sklearn.metrics import f1_score, zero_one_loss
import itertools

def create_groupkfolds(df, n_folds, group_col):
    df['kfold'] = -1
    y = df.root
    groups = df[group_col]
    
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for f, (t_, v_) in enumerate(gkf.split(X=df, y=y, groups=groups)):
        df.loc[v_, 'kfold'] = f
         
    return df

def evaluate_weights(df, fold, model_configs):
    """
    Evaluate ensemble with given weights for a specific fold
    """
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Create feature pipeline
    feature_pipeline = Pipeline(steps=[
        ("Language Features", LanguageFeature()),
        ("Graph Features", GraphFeatures()),
        ("Node Features", NodeFeatures()),
        ("Dataset Creation", FormatDataFrame()),
        ("Language One Hot Encoding", LanguageOHE(
            enc_lan=f"{model_configs[0]['model_name']}/lan_encoder_{model_configs[0]['model_name']}_{fold}.pkl",
            enc_lan_family=f"{model_configs[0]['model_name']}/lan_family_encoder_{model_configs[0]['model_name']}_{fold}.pkl"
        ))
    ])

    # Process data
    train_data = feature_pipeline.fit_transform(df_train)
    valid_data = feature_pipeline.transform(df_valid)
    
    x_train = train_data.drop(columns=config.TRAIN_DROP_COLS)
    y_train = train_data.is_root.values
    x_valid = valid_data.drop(columns=config.TRAIN_DROP_COLS)
    y_valid = valid_data.is_root.values

    # Initialize arrays to store predictions
    valid_preds = np.zeros((len(x_valid), len(model_configs)))
    valid_probs = np.zeros((len(x_valid), len(model_configs)))

    # Get predictions from each model
    for i, m_config in enumerate(model_configs):
        model_name = m_config['model_name']
        model_path = os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, f'{model_name}/{model_name}_{fold}.pkl')
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return None, None
        
        clf = joblib.load(model_path)
        valid_preds[:, i] = clf.predict(x_valid)
        valid_probs[:, i] = clf.predict_proba(x_valid)[:, 1]

    # Calculate weighted predictions
    weights = np.array([m_config['weight'] for m_config in model_configs])
    weights = weights / np.sum(weights)  # Normalize weights
    
    weighted_probs = np.sum(valid_probs * weights, axis=1)
    weighted_preds = (weighted_probs > 0.5).astype(int)

    # Get max probability predictions for each sentence
    valid_data['prediction_probability'] = weighted_probs
    valid_max_rows = valid_data.loc[valid_data.groupby(by=['sentence', 'language'])['prediction_probability'].idxmax()]
    valid_result = valid_max_rows[['sentence', 'language', 'node_number']]
    valid_result = valid_result.rename(columns={'node_number': 'predicted_root'})

    # Merge predictions back to original dataframe
    df_valid = pd.merge(df_valid, valid_result, on=['sentence', 'language'], how='inner')

    # Calculate metrics
    valid_f1 = f1_score(y_valid, weighted_preds, average='macro')
    valid_zero_one = zero_one_loss(df_valid['root'], df_valid['predicted_root'])

    return valid_f1, valid_zero_one

def tune_weights(model_configs, n_folds=10, weight_step=0.1):
    """
    Tune weights for ensemble models using grid search
    """
    print("Loading data...")
    df = pd.read_csv(config.TRAINING_DATA_PATH)
    df = create_groupkfolds(df, n_folds, 'sentence')
    
    # Generate weight combinations
    n_models = len(model_configs)
    weight_range = np.arange(0.1, 1.0, weight_step)
    weight_combinations = []
    
    for weights in itertools.product(weight_range, repeat=n_models-1):
        # Calculate last weight to ensure sum = 1
        last_weight = 1 - sum(weights)
        if last_weight >= 0.1:  # Only include if last weight is reasonable
            weight_combinations.append(weights + (last_weight,))
    
    print(f"Testing {len(weight_combinations)} weight combinations...")
    
    # Store results
    results = []
    
    for weights in weight_combinations:
        # Update model configs with current weights
        for i, m_config in enumerate(model_configs):
            m_config['weight'] = weights[i]
        
        # Evaluate across folds
        fold_f1_scores = []
        fold_zero_one_scores = []
        
        for fold in range(n_folds):
            f1, zero_one = evaluate_weights(df, fold, model_configs)
            if f1 is not None:
                fold_f1_scores.append(f1)
                fold_zero_one_scores.append(zero_one)
        
        if fold_f1_scores:  # If we got valid results
            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)
            mean_zero_one = np.mean(fold_zero_one_scores)
            std_zero_one = np.std(fold_zero_one_scores)
            
            results.append({
                'weights': weights,
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'mean_zero_one': mean_zero_one,
                'std_zero_one': std_zero_one
            })
            
            print(f"Weights: {weights}")
            print(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
            print(f"Zero-One Loss: {mean_zero_one:.4f} ± {std_zero_one:.4f}")
            print("-" * 50)
    
    # Sort results by F1 score
    results.sort(key=lambda x: x['mean_f1'], reverse=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../results/ensemble_weight_tuning_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("Ensemble Weight Tuning Results\n")
        f.write("=" * 50 + "\n\n")
        
        for i, result in enumerate(results[:10], 1):  # Save top 10 results
            f.write(f"Rank {i}\n")
            f.write(f"Weights: {result['weights']}\n")
            f.write(f"F1 Score: {result['mean_f1']:.4f} ± {result['std_f1']:.4f}\n")
            f.write(f"Zero-One Loss: {result['mean_zero_one']:.4f} ± {result['std_zero_one']:.4f}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Return best weights
    best_result = results[0]
    print("\nBest weights found:")
    print(f"Weights: {best_result['weights']}")
    print(f"F1 Score: {best_result['mean_f1']:.4f} ± {best_result['std_f1']:.4f}")
    print(f"Zero-One Loss: {best_result['mean_zero_one']:.4f} ± {best_result['std_zero_one']:.4f}")
    
    return best_result['weights']

if __name__ == "__main__":
    # Define models to ensemble
    model_configs = [
        {'model_name': 'lgbm_logloss', 'weight': 0.5},
        {'model_name': 'lgbm_zero_one_loss', 'weight': 0.5}
    ]
    
    # Tune weights
    best_weights = tune_weights(model_configs, n_folds=10, weight_step=0.1)
    
    # Print final model configuration
    print("\nFinal model configuration:")
    for i, m_config in enumerate(model_configs):
        print(f"{m_config['model_name']}: {best_weights[i]:.2f}") 