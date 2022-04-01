# Balanced-Evolutionary-Semi-Stacking

* Required Python 3 packages: 
    1. sklearn (https://github.com/scikit-learn/scikit-learn)
    2. imblearn (https://github.com/scikit-learn-contrib/imbalanced-learn)
    3. lightgbm (optional, https://github.com/microsoft/LightGBM)
    4. xgboost (optional, https://github.com/dmlc/xgboost)

* ESAE is compatible with most sklearn APIs but is not strictly tested.

* Import: `from ESAE import EvolutionarySAE`

* Train: `fit(X, y)`, with target `0, ..., K` as the labels

* Predict: `predict(X)`
