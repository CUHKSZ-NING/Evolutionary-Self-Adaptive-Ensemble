# Forest of Evolutionary Hierarchical Classifiers (FEHC)

* Code for manuscript "To Combat Multi-class Imbalanced Problems by Aggregating Evolutionary Hierarchical Classifiers" (In Submission)

* Required Python 3 packages: 
    1. sklearn (https://github.com/scikit-learn/scikit-learn)
    2. imblearn (https://github.com/scikit-learn-contrib/imbalanced-learn)
    3. lightgbm (optional, https://github.com/microsoft/LightGBM)
    4. xgboost (optional, https://github.com/dmlc/xgboost)

* FEHC is compatible with most sklearn APIs but is not strictly tested.

* Import: `from FEHCClassifier import FEHCClassifier`

* Train: `fit(X, y)`, with target `0, ..., K` as the labels.

* Predict: `predict(X)` (hard), `predict_proba(X)` (probalistic), or `predict(X, n_estimator=1)` (using the EHMC instead of ESAE to predict, faster but possibly leading to performance degradation).
