from sklearn.naive_bayes import MultinomialNB, GaussianNB
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

models = {

    'mnb':MultinomialNB(class_prior=[0.5, 0.5]),
    'lgbm':lgb.LGBMClassifier(n_estimator=100, verbose=-1,random_state=42,is_unbalance=True),
    'lr':LogisticRegression(random_state=42,penalty='l2',C=0.0001),
    'xgb':XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=1, 
        random_state=42,
        verbose=0
    ),
    'rf':RandomForestClassifier(random_state=42),
    'lgbm_other':lgb.LGBMClassifier(n_estimator=100, verbose=-1,random_state=42),
    'lgbm_jp':lgb.LGBMClassifier(n_estimator=100, verbose=-1,random_state=42),
    'lda':LinearDiscriminantAnalysis(priors=[0.5, 0.5]),
    'qda':QuadraticDiscriminantAnalysis(),
    
    # Best model with AUC metric
    'xgb_auc': XGBClassifier(
        booster='gbtree',
        lambda_=1.1199566483339617e-07,
        alpha=6.101403004331885e-07,
        learning_rate=0.14721823854055904,
        n_estimators=961,
        max_depth=9,
        min_child_weight=5,
        gamma=0.0002965398395149084,
        subsample=0.9820117486294455,
        colsample_bytree=0.6166351578002189,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        verbose=0
    ),
    
    # Best model with log loss metric
    'xgb_logloss': XGBClassifier(
        booster='dart',
        lambda_=0.00044774351548967705,
        alpha=0.000694912939521992,
        learning_rate=0.13821190677129985,
        n_estimators=163,
        max_depth=9,
        min_child_weight=6,
        gamma=1.6177472372323167e-05,
        subsample=0.9519836350509681,
        colsample_bytree=0.9424162725030374,
        sample_type='weighted',
        normalize_type='tree',
        rate_drop=0.001911472383359846,
        skip_drop=0.0010850466610653905,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        verbose=0
    ),
    
    # Best LightGBM model with log loss metric
    'lgbm_logloss': lgb.LGBMClassifier(
        boosting_type='dart',
        num_leaves=100,
        learning_rate=0.2404304987820357,
        n_estimators=993,
        max_depth=8,
        min_child_samples=98,
        subsample=0.7758415112715576,
        colsample_bytree=0.8449875451391139,
        reg_alpha=8.410243613130436e-08,
        reg_lambda=1.107002443010181e-08,
        drop_rate=0.17665455447126532,
        skip_drop=0.3917329494953023,
        objective='binary',
        metric='binary_logloss',
        random_state=42,
        verbose=-1,
        is_unbalance=True
    ),

    # Best model from threshold optimization
    'xgb_zero_one_loss': XGBClassifier(
        booster='gbtree',
        lambda_=0.01646184064610464,
        alpha=8.925449429850201e-07,
        learning_rate=0.04766323397168665,
        n_estimators=726,
        max_depth=8,
        min_child_weight=5,
        gamma=1.2053850658756302e-08,
        subsample=0.9816077266525208,
        colsample_bytree=0.8483455947091667,
        objective='binary:logistic',
        random_state=42,
        verbose=0
    ),

    # Best LightGBM model from tuning
    'lgbm_zero_one_loss': lgb.LGBMClassifier(
        boosting_type='dart',
        num_leaves=90,
        learning_rate=0.074831705371424,
        n_estimators=967,
        max_depth=6,
        min_child_samples=63,
        subsample=0.822129055301889,
        colsample_bytree=0.769131350148678,
        reg_alpha=0.00013658550129263758,
        reg_lambda=1.6283011289438004e-07,
        drop_rate=0.14187438966796803,
        skip_drop=0.4876667967884481,
        objective='binary',
        metric='binary_logloss',
        random_state=42,
        verbose=-1,
        is_unbalance=True
    )
    
}