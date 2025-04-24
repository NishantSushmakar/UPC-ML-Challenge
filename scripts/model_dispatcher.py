from sklearn.naive_bayes import MultinomialNB, GaussianNB
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


models = {

    'mnb':MultinomialNB(),
    'lgbm':lgb.LGBMClassifier(n_estimator=1000, verbose=-1,random_state=42),
    'lr':LogisticRegression(random_state=42)

}