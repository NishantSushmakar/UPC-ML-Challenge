from sklearn.naive_bayes import MultinomialNB, GaussianNB
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class SentenceAwareRandomForest(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None, meta=None):
        # Drop 'sentence' and 'language' for model input
        XX = X.drop(['sentence', 'language'], axis=1)
        self._X = X.reset_index(drop=True)
        self._y = np.array(y)
        self._groups = self._X[['sentence', 'language']].drop_duplicates().reset_index(drop=True)

        # Define self.estimator_ before calling _make_estimator
        self.estimator_ = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter="best",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state
        )

        self.estimators_ = []
        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_estimators):
            sampled_groups = resample(self._groups, replace=True, random_state=rng.randint(0, 1e6))
            merged = self._X.merge(sampled_groups, on=['sentence', 'language'], how='inner')
            indices = self._X.index.get_indexer(merged.index)

            X_sample = self._X.drop(['sentence', 'language'], axis=1).iloc[indices]
            y_sample = self._y[indices]

            estimator = self._make_estimator(append=True)
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)

        # super().fit(XX, y, sample_weight)
        return self


    def predict_proba(self, X):
        XX = X.copy()
        for col in ['sentence', 'language']:
            if col in XX.columns:
                XX = XX.drop(col, axis=1)

        all_probas = np.array([est.predict_proba(XX) for est in self.estimators_])
        avg_probas = np.mean(all_probas, axis=0)
        return avg_probas

    def predict(self, X):
        avg_probas = self.predict_proba(X)
        return np.argmax(avg_probas, axis=1)
    

models = {

    'mnb':MultinomialNB(),
    'lgbm':lgb.LGBMClassifier(n_estimator=1000, verbose=-1,random_state=42),
    'lr':LogisticRegression(random_state=42,max_iter=1000),
    'rf': RandomForestClassifier(
        # oob_score=True,
        n_estimators=100,          # Number of trees
        class_weight='balanced',    # Handle imbalanced classes
        max_depth=8,              # Control overfitting
        min_samples_leaf=5,        # Prevent overly complex trees
    ),
    'rff': SentenceAwareRandomForest(
        oob_score=True,
        n_estimators=100,
        class_weight='balanced',
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    ),
    'xgb': XGBClassifier(
        n_estimators=1000,
        random_state=42
    ),
    'gnb': GaussianNB(),
    'gb': GradientBoostingClassifier()

}