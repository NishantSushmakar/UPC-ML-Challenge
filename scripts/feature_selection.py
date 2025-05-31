import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss
import lightgbm as lgb
from feature_creation import *
import config
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

class FeatureSelector:
    """
    A class for feature selection using importance threshold method.
    """
    
    def __init__(self, model, n_folds=5, random_state=42):
        """
        Initialize the FeatureSelector
        
        Parameters:
        -----------
        model : estimator object
            The base model to use for feature selection
        n_folds : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.model = model
        self.n_folds = n_folds
        self.random_state = random_state
        self.feature_pipeline = None
        self.feature_importances = {}
        
    def _create_folds(self, df):
        """
        Create cross-validation folds based on groups
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added kfold column
        """
        df = df.copy()
        df['kfold'] = -1
        y = df.root
        groups = df['sentence']
        
        gkf = GroupKFold(n_splits=self.n_folds)
        
        for f, (t_, v_) in enumerate(gkf.split(X=df, y=y, groups=groups)):
            df.loc[v_, 'kfold'] = f
            
        return df
    
    def _prepare_pipeline(self):
        """
        Create the feature processing pipeline
        
        Returns:
        --------
        Pipeline
            The feature processing pipeline
        """
        return Pipeline(steps=[
            ("Language Features", LanguageFeature()),
            ("Graph Features", GraphFeatures()),
            ("Node Features", NodeFeatures()),
            ("Dataset Creation", FormatDataFrame()),
            ("Language One Hot Encoding", LanguageOHE(
                enc_lan="lgbm/lan_encoder_lgbm.pkl",
                enc_lan_family="lgbm/lan_family_encoder_lgbm.pkl"
            ))
        ])
    
    def zero_one_loss(self, y_true, y_pred):
        """
        Calculate zero-one loss
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
            
        Returns:
        --------
        float
            Zero-one loss value
        """
        if len(y_true) != len(y_pred):
            raise ValueError("The lengths of true and predicted labels must match.")
        incorrect = sum(yt != yp for yt, yp in zip(y_true, y_pred))
        return incorrect / len(y_true)
    
    def evaluate_features(self, df, features):
        """
        Optimized feature evaluation using cross-validation
        """
        df = self._create_folds(df)
        fold_scores = []
        all_metrics = {
            'zero_one_loss': [],
            'f1': [],
            'auc': []
        }
        
        # Store feature importance across folds
        feature_importances = {feature: 0 for feature in features}
        
        # Process data once for all folds
        if self.feature_pipeline is None:
            self.feature_pipeline = self._prepare_pipeline()
            processed_data = self.feature_pipeline.fit_transform(df)
        else:
            processed_data = self.feature_pipeline.transform(df)
        
        # Ensure we have the required columns
        required_cols = ['sentence', 'language', 'node_number', 'is_root']
        missing_cols = [col for col in required_cols if col not in processed_data.columns]
        if missing_cols:
            # Add missing columns from original dataframe
            for col in missing_cols:
                processed_data[col] = df[col].values
        
        # Normalize features once
        scaler = MinMaxScaler()
        x_all = scaler.fit_transform(processed_data[features])
        x_all = pd.DataFrame(x_all, columns=features)
        y_all = processed_data.is_root.values
        
        for fold in range(self.n_folds):
            # Split data
            train_idx = df[df.kfold != fold].index
            valid_idx = df[df.kfold == fold].index
            
            x_train = x_all.iloc[train_idx]
            y_train = y_all[train_idx]
            x_valid = x_all.iloc[valid_idx]
            y_valid = y_all[valid_idx]
            
            # Train model
            clf = self.model.fit(x_train, y_train)
            
            # Get predictions
            y_valid_pred = clf.predict(x_valid)
            y_valid_proba = clf.predict_proba(x_valid)[:, 1]
            
            # Calculate metrics
            all_metrics['f1'].append(f1_score(y_valid, y_valid_pred))
            all_metrics['auc'].append(roc_auc_score(y_valid, y_valid_proba))
            
            # Update feature importance
            if hasattr(clf, 'feature_importances_'):
                for i, feature in enumerate(features):
                    feature_importances[feature] += clf.feature_importances_[i] / self.n_folds
            
            # Calculate zero-one loss
            valid_data = processed_data.iloc[valid_idx].copy()
            valid_data['prediction_probability'] = y_valid_proba
            
            # Group by sentence and language, get max probability row
            valid_max_rows = valid_data.loc[valid_data.groupby(['sentence', 'language'])['prediction_probability'].idxmax()]
            
            # Create result dataframe with required columns
            valid_result = valid_max_rows[['sentence', 'language', 'node_number']].copy()
            valid_result = valid_result.rename(columns={'node_number': 'predicted_root'})
            
            # Merge with original data to get true roots
            valid_data = pd.merge(valid_data, valid_result, on=['sentence', 'language'], how='inner')
            valid_zero_one_loss = self.zero_one_loss(valid_data['is_root'], valid_data['predicted_root'])
            
            all_metrics['zero_one_loss'].append(valid_zero_one_loss)
            fold_scores.append(valid_zero_one_loss)
        
        # Update feature importances
        self.feature_importances = feature_importances
        
        # Calculate mean of all metrics
        mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        
        return np.mean(fold_scores), mean_metrics
    
    def importance_threshold_selection(self, df, initial_features, threshold_percentile=20, variance_threshold=0.01, correlation_threshold=0.95):
        """
        Fast feature selection based on importance threshold, variance, and correlation
        """
        print("Starting feature selection process...")
        print(f"Initial number of features: {len(initial_features)}")
        
        # Initialize progress tracking
        progress = {
            'step': 1,
            'total_steps': 3,
            'features_removed': [],
            'metrics': [],
            'time_taken': []
        }
        
        # Process data once
        if self.feature_pipeline is None:
            self.feature_pipeline = self._prepare_pipeline()
        
        print("\nProcessing data through feature pipeline...")
        start_time = time.time()
        train_data = self.feature_pipeline.fit_transform(df)
        x_train = train_data[initial_features]
        y_train = train_data.is_root.values
        progress['time_taken'].append(time.time() - start_time)
        
        # 1. Remove low variance features
        print(f"\nStep {progress['step']}/{progress['total_steps']}: Removing low variance features...")
        start_time = time.time()
        
        variances = x_train.var()
        high_variance_features = variances[variances > variance_threshold].index.tolist()
        removed_variance = len(initial_features) - len(high_variance_features)
        progress['features_removed'].append({
            'step': 'Variance Filtering',
            'removed': removed_variance,
            'remaining': len(high_variance_features),
            'threshold': variance_threshold
        })
        print(f"Removed {removed_variance} low variance features")
        print(f"Features remaining: {len(high_variance_features)}")
        progress['time_taken'].append(time.time() - start_time)
        progress['step'] += 1
        
        # 2. Remove highly correlated features
        print(f"\nStep {progress['step']}/{progress['total_steps']}: Removing highly correlated features...")
        start_time = time.time()
        
        print("Calculating correlation matrix...")
        correlation_matrix = x_train[high_variance_features].corr().abs()
        
        print("Identifying highly correlated features...")
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        uncorrelated_features = [f for f in high_variance_features if f not in to_drop]
        
        removed_correlation = len(high_variance_features) - len(uncorrelated_features)
        progress['features_removed'].append({
            'step': 'Correlation Filtering',
            'removed': removed_correlation,
            'remaining': len(uncorrelated_features),
            'threshold': correlation_threshold
        })
        print(f"Removed {removed_correlation} highly correlated features")
        print(f"Features remaining: {len(uncorrelated_features)}")
        progress['time_taken'].append(time.time() - start_time)
        progress['step'] += 1
        
        # 3. Feature importance selection
        print(f"\nStep {progress['step']}/{progress['total_steps']}: Selecting features based on importance...")
        start_time = time.time()
        
        x_train_filtered = x_train[uncorrelated_features]
        
        print("Training model for feature importance...")
        self.model.fit(x_train_filtered, y_train)
        
        if not hasattr(self.model, 'feature_importances_'):
            print("Model doesn't have feature_importances_ attribute. Cannot perform importance threshold selection.")
            return uncorrelated_features, float('inf')
        
        importance = self.model.feature_importances_
        
        # Create dataframe with feature importances
        feature_importance_df = pd.DataFrame({
            'feature': uncorrelated_features,
            'importance': importance,
            'variance': variances[uncorrelated_features]
        }).sort_values(by='importance', ascending=False)
        
        # Get threshold value
        threshold = np.percentile(importance, 100 - threshold_percentile)
        
        # Select features above threshold
        selected_features = feature_importance_df[feature_importance_df['importance'] >= threshold]['feature'].tolist()
        
        removed_importance = len(uncorrelated_features) - len(selected_features)
        progress['features_removed'].append({
            'step': 'Importance Selection',
            'removed': removed_importance,
            'remaining': len(selected_features),
            'threshold': f"{threshold_percentile}th percentile"
        })
        progress['time_taken'].append(time.time() - start_time)
        
        # Evaluate selected features
        print("\nEvaluating selected features...")
        score, metrics = self.evaluate_features(df, selected_features)
        progress['metrics'].append({
            'zero_one_loss': score,
            'f1_score': metrics['f1'],
            'auc': metrics['auc']
        })
        
        # Print detailed progress report
        self._print_progress_report(progress, feature_importance_df)
        
        # Plot feature importance
        self._plot_feature_importance(feature_importance_df, threshold)
        
        # Plot correlation matrix of selected features
        self._plot_correlation_matrix(x_train[selected_features])
        
        return selected_features, score
    
    def _print_progress_report(self, progress, feature_importance_df):
        """
        Print detailed progress report of feature selection process
        """
        print("\n" + "="*50)
        print("FEATURE SELECTION PROGRESS REPORT")
        print("="*50)
        
        # Print step-by-step progress
        print("\nStep-by-Step Progress:")
        print("-"*30)
        for i, step in enumerate(progress['features_removed']):
            print(f"\nStep {i+1}: {step['step']}")
            print(f"  - Features removed: {step['removed']}")
            print(f"  - Features remaining: {step['remaining']}")
            print(f"  - Threshold: {step['threshold']}")
            print(f"  - Time taken: {progress['time_taken'][i]:.2f} seconds")
        
        # Print final metrics
        print("\nFinal Performance Metrics:")
        print("-"*30)
        metrics = progress['metrics'][-1]
        print(f"Zero-one loss: {metrics['zero_one_loss']:.4f}")
        print(f"F1 score: {metrics['f1_score']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        
        # Print top features
        print("\nTop 10 Most Important Features:")
        print("-"*30)
        top_features = feature_importance_df.head(10)
        for _, row in top_features.iterrows():
            print(f"- {row['feature']}:")
            print(f"  Importance: {row['importance']:.4f}")
            print(f"  Variance: {row['variance']:.4f}")
        
        # Print total time
        total_time = sum(progress['time_taken'])
        print("\nTotal Time:")
        print("-"*30)
        print(f"Total processing time: {total_time:.2f} seconds")
        
        print("\n" + "="*50)
    
    def _plot_feature_importance(self, feature_importance_df, threshold=None):
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_importance_df : pandas.DataFrame
            DataFrame with feature importances
        threshold : float, default=None
            Threshold value for feature selection
        """
        plt.figure(figsize=(12, 10))
        
        # Plot feature importance
        top_features = feature_importance_df.head(30)  # Show top 30 features
        
        sns.barplot(x='importance', y='feature', data=top_features)
        
        if threshold is not None:
            plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
            
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.legend()
        plt.tight_layout()
        plt.savefig('../results/feature_importance.png')
        plt.close()

    def _plot_correlation_matrix(self, data):
        """
        Plot correlation matrix of selected features
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing selected features
        """
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot correlation matrix
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    linewidths=.5,
                    cbar_kws={"shrink": .5})
        
        plt.title('Correlation Matrix of Selected Features')
        plt.tight_layout()
        plt.savefig('../results/feature_correlation_matrix.png')
        plt.close()


def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv(config.TRAINING_DATA_PATH)
    
    # Create output directories if they don't exist
    os.makedirs('../results', exist_ok=True)
    
    # Initialize base model with faster parameters
    base_model = lgb.LGBMClassifier(
        n_estimators=100,  # Reduced number of trees
        verbose=-1,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1        # Use all CPU cores
    )
    
    # Initialize feature selector
    selector = FeatureSelector(model=base_model, n_folds=5)
    
    # Get initial feature set
    feature_pipeline = selector._prepare_pipeline()
    
    # Get initial features
    print("Processing initial features...")
    train_data = feature_pipeline.fit_transform(df)
    initial_features = [col for col in train_data.columns if col not in config.TRAIN_DROP_COLS]
    
    print(f"Initial feature set contains {len(initial_features)} features")
    
    # Use importance threshold selection with variance and correlation filtering
    print("\n---- Feature Selection: Importance Threshold with Variance and Correlation Analysis ----")
    selected_features, score = selector.importance_threshold_selection(
        df, 
        initial_features, 
        threshold_percentile=20,
        variance_threshold=0.01,    # Remove features with variance < 0.01
        correlation_threshold=0.95   # Remove features with correlation > 0.95
    )
    
    # Final evaluation
    print("\n---- Final Evaluation ----")
    final_score, final_metrics = selector.evaluate_features(df, selected_features)
    
    print(f"Final zero-one loss: {final_score:.4f}")
    print(f"Final F1 score: {final_metrics['f1']:.4f}")
    print(f"Final AUC: {final_metrics['auc']:.4f}")
    
    # Save results
    results = {
        'selected_features': selected_features,
        'best_score': score,
        'final_score': final_score,
        'final_metrics': final_metrics
    }
    
    joblib.dump(results, '../resources/feature_selection_results.pkl')
    
    # Print final selected features
    print("\nFinal Selected Features:")
    for feature in selected_features:
        print(f"- {feature}")


if __name__ == "__main__":
    main()
