# Forest of Evolutionary Hierarchical Classifiers (FEHC)

* Code for the manuscript "To Combat Multi-class Imbalanced Problems by Aggregating Evolutionary Hierarchical Classifiers" (In Submission)

* Required Python 3 packages: 
    1. sklearn (https://github.com/scikit-learn/scikit-learn)
    2. glob (for dataset loading, `datasets = joblib.load('MCIDatasets.pkl')`)

* FEHC is compatible with most sklearn APIs but is not strictly tested.

* Import: `from FEHCClassifier import FEHCClassifier`

* Train: `fit(X, y)`, with target $y_i \in \{0, ..., K\}$ as the labels.

* Predict: `predict(X)` (hard prediction), `predict_proba(X)` (probalistic prediction), or `predict(X, n_estimator=1)` (using the EHMC instead of ESAE to predict, faster but possibly leading to performance degradation).

* Non-trivital parameters: 
    1. `base_estimators`: dict, `default={'DT': DecisionTreeClassifier()}`, candidate classifier set $\mathcal{C}$, should have predict_proba() function"
    2. `n_estimators`: int, `default=30`, "the number of EHMCs $n$ in the FEHC"
    3. `population`: int, `default=10`, "the population size $\theta_P$ of the MCGA"
    4. `iteration`: int, `default=5`, "the number of iteration rounds $\theta_I$ of the MCGA"
