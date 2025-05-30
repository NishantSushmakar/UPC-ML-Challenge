import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import config
from feature_creation import *
import joblib
import os
from datetime import datetime
from sklearn.metrics import f1_score, zero_one_loss, precision_score, recall_score, roc_auc_score, log_loss

def create_groupkfolds(df, n_folds, group_col):
    df['kfold'] = -1
    y = df.root
    groups = df[group_col]
    
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for f, (t_, v_) in enumerate(gkf.split(X=df, y=y, groups=groups)):
        df.loc[v_, 'kfold'] = f
         
    return df

def get_model_predictions(df, fold, model_names):
    """
    Get predictions from each model for a specific fold
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
            enc_lan=f"{model_names[0]}/lan_encoder_{model_names[0]}_{fold}.pkl",
            enc_lan_family=f"{model_names[0]}/lan_family_encoder_{model_names[0]}_{fold}.pkl"
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
    train_probs = np.zeros((len(x_train), len(model_names)))
    valid_probs = np.zeros((len(x_valid), len(model_names)))

    # Get predictions from each model
    for i, model_name in enumerate(model_names):
        model_path = os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, f'{model_name}/{model_name}_{fold}.pkl')
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return None, None, None, None
        
        clf = joblib.load(model_path)
        train_probs[:, i] = clf.predict_proba(x_train)[:, 1]
        valid_probs[:, i] = clf.predict_proba(x_valid)[:, 1]

    return train_probs, y_train, valid_probs, y_valid, valid_data, df_valid

def train_meta_model(model_names, n_folds=10, meta_model_params=None):
    """
    Train a logistic regression meta-model on top of base models' probabilities
    """
    if meta_model_params is None:
        meta_model_params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        }

    print("Loading data...")
    df = pd.read_csv(config.TRAINING_DATA_PATH)
    df = create_groupkfolds(df, n_folds, 'sentence')
    
    # Store results for each fold
    fold_results = []
    
    for fold in range(n_folds):
        print(f"\nProcessing fold {fold}")
        
        # Get predictions from base models
        train_probs, y_train, valid_probs, y_valid, valid_data, df_valid = get_model_predictions(df, fold, model_names)
        
        if train_probs is None:
            continue
        
        # Train meta-model
        meta_model = LogisticRegression(**meta_model_params)
        meta_model.fit(train_probs, y_train)
        
        # Get meta-model predictions
        valid_meta_probs = meta_model.predict_proba(valid_probs)[:, 1]
        valid_meta_preds = (valid_meta_probs > 0.5).astype(int)
        
        # Get max probability predictions for each sentence
        valid_data['prediction_probability'] = valid_meta_probs
        valid_max_rows = valid_data.loc[valid_data.groupby(by=['sentence', 'language'])['prediction_probability'].idxmax()]
        valid_result = valid_max_rows[['sentence', 'language', 'node_number']]
        valid_result = valid_result.rename(columns={'node_number': 'predicted_root'})
        
        # Merge predictions back to original dataframe
        df_valid = pd.merge(df_valid, valid_result, on=['sentence', 'language'], how='inner')
        
        # Calculate metrics
        metrics = {
            'precision': precision_score(y_valid, valid_meta_preds, average='macro'),
            'recall': recall_score(y_valid, valid_meta_preds, average='macro'),
            'f1': f1_score(y_valid, valid_meta_preds, average='macro'),
            'auc': roc_auc_score(y_valid, valid_meta_probs),
            'zero_one_loss': zero_one_loss(df_valid['root'], df_valid['predicted_root']),
            'log_loss': log_loss(y_valid, valid_meta_probs)
        }
        
        fold_results.append(metrics)
        
        # Save meta-model for this fold
        meta_model_dir = os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, 'meta_model')
        if not os.path.exists(meta_model_dir):
            os.makedirs(meta_model_dir)
        joblib.dump(meta_model, os.path.join(meta_model_dir, f'meta_model_{fold}.pkl'))
        
        print(f"Fold {fold} results:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Zero-One Loss: {metrics['zero_one_loss']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Log Loss: {metrics['log_loss']:.4f}")
    
    # Calculate average metrics
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_results])
        for metric in ['precision', 'recall', 'f1', 'auc', 'zero_one_loss', 'log_loss']
    }
    
    std_metrics = {
        metric: np.std([fold[metric] for fold in fold_results])
        for metric in ['precision', 'recall', 'f1', 'auc', 'zero_one_loss', 'log_loss']
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../results/meta_model_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("Meta-Model Training Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Average Metrics Across Folds:\n")
        f.write(f"Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}\n")
        f.write(f"Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}\n")
        f.write(f"AUC: {avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}\n")
        f.write(f"Zero-One Loss: {avg_metrics['zero_one_loss']:.4f} ± {std_metrics['zero_one_loss']:.4f}\n")
        f.write(f"Log Loss: {avg_metrics['log_loss']:.4f} ± {std_metrics['log_loss']:.4f}\n")
        
        f.write("\nPer-fold results:\n")
        for i, fold_metrics in enumerate(fold_results):
            f.write(f"\nFold {i}:\n")
            for metric, value in fold_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
    print("\nAverage metrics across folds:")
    print(f"Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"F1 Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"AUC: {avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
    print(f"Zero-One Loss: {avg_metrics['zero_one_loss']:.4f} ± {std_metrics['zero_one_loss']:.4f}")
    print(f"Log Loss: {avg_metrics['log_loss']:.4f} ± {std_metrics['log_loss']:.4f}")

if __name__ == "__main__":
    # Define base models to use
    model_names = ['lgbm_logloss', 'lgbm_zero_one_loss']
    
    # Train meta-model
    train_meta_model(model_names, n_folds=10)

    model_names = ['lgbm_logloss', 'lgbm_zero_one_loss', 'xgb_zero_one_loss']
    train_meta_model(model_names, n_folds=10)

    