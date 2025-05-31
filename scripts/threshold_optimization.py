import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from model_dispatcher import models
import config
from sklearn.model_selection import StratifiedGroupKFold
import joblib
import os
from datetime import datetime
from feature_creation import *

def evaluate_model(model_name, n_folds=5):
    """
    Evaluate model performance using cross-validation.
    
    Parameters:
    - model_name: Name of the model to evaluate
    - n_folds: Number of folds for cross-validation
    
    Returns:
    - mean_f1_score: Average F1 score across folds
    - std_f1_score: Standard deviation of F1 scores
    - fold_scores: List of F1 scores for each fold
    """
    # Load data
    df = pd.read_csv(config.TRAINING_DATA_PATH)
    
    # Prepare groups (using sentence as group)
    groups = df['sentence']
    
    # Initialize cross-validation
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store scores for each fold
    fold_scores = []
    threshold_scores = {}
    
    # Initialize threshold_scores dictionary
    for threshold in np.arange(0.1, 0.9, 0.05):
        threshold_scores[threshold] = []
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv.split(df, df['root'], groups)):
        print(f"\nProcessing fold {fold + 1}/{n_folds}")
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_valid = df.iloc[val_idx].reset_index(drop=True)
        
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
        
        # Process data through feature pipeline
        train_data = feature_pipeline.fit_transform(df_train)
        valid_data = feature_pipeline.transform(df_valid)
        
        # Prepare features and target
        X_train = train_data.drop(columns=config.TRAIN_DROP_COLS)
        y_train = train_data.is_root.values
        X_valid = valid_data.drop(columns=config.TRAIN_DROP_COLS)
        y_valid = valid_data.is_root.values
        
        # Train model
        model = models[model_name]
        model.fit(X_train, y_train)
        
        # Get predictions
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        
        # Add prediction probabilities to validation dataframe
        valid_data['prediction_probability'] = y_valid_proba
        
        # Evaluate different thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            # Convert probabilities to binary predictions
            y_pred = (y_valid_proba >= threshold).astype(int)
            score = f1_score(y_valid, y_pred, average='macro')
            threshold_scores[threshold].append(score)
            
            print(f"Threshold {threshold:.2f} - F1 Score: {score:.4f}")
    
    # Calculate mean scores for each threshold
    mean_scores = {t: np.mean(scores) for t, scores in threshold_scores.items()}
    
    # Find best threshold and all thresholds within 0.01 of the best score
    best_threshold = max(mean_scores.items(), key=lambda x: x[1])
    best_score = best_threshold[1]
    best_thresholds = [t for t, score in mean_scores.items() if abs(score - best_score) <= 0.01]
    
    return best_thresholds, threshold_scores

def plot_threshold_analysis(threshold_scores, best_thresholds):
    """
    Plot threshold analysis results.
    
    Parameters:
    - threshold_scores: Dictionary containing scores for all thresholds
    - best_thresholds: List of thresholds that performed best
    """
    plt.figure(figsize=(10, 6))
    thresholds = list(threshold_scores.keys())
    mean_scores = [np.mean(scores) for scores in threshold_scores.values()]
    
    plt.plot(thresholds, mean_scores, 'b-', label='Mean F1 Score')
    
    # Plot best thresholds
    best_scores = [np.mean(threshold_scores[t]) for t in best_thresholds]
    plt.scatter(best_thresholds, best_scores, color='red', s=100, 
               label=f'Best Thresholds: {", ".join([f"{t:.2f}" for t in best_thresholds])}')
    
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Threshold Analysis')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'../results/threshold_analysis_{timestamp}.png')
    plt.close()

def main():
    # Model to evaluate
    model_name = 'lgbm_logloss'
    
    # Create directory for model resources if it doesn't exist
    if not os.path.exists(f'../resources/{model_name}'):
        os.makedirs(f'../resources/{model_name}')
    
    # Evaluate model
    print(f"\nEvaluating {model_name}...")
    best_thresholds, threshold_scores = evaluate_model(model_name)
    
    print("\nResults:")
    print(f"Best thresholds: {best_thresholds}")
    print(f"Best F1 score: {np.mean(threshold_scores[best_thresholds[0]]):.4f}")
    
    # Plot threshold analysis
    plot_threshold_analysis(threshold_scores, best_thresholds)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'../results/threshold_analysis_{timestamp}.txt', 'w') as f:
        f.write(f"Model: {model_name}\n\n")
        f.write("Threshold Analysis Results:\n")
        for threshold in sorted(threshold_scores.keys()):
            mean_score = np.mean(threshold_scores[threshold])
            std_score = np.std(threshold_scores[threshold])
            f.write(f"Threshold {threshold:.2f}: {mean_score:.4f} ± {std_score:.4f}\n")
        
        f.write("\nBest Thresholds:\n")
        for threshold in best_thresholds:
            mean_score = np.mean(threshold_scores[threshold])
            std_score = np.std(threshold_scores[threshold])
            f.write(f"Threshold {threshold:.2f}: {mean_score:.4f} ± {std_score:.4f}\n")

if __name__ == "__main__":
    main() 