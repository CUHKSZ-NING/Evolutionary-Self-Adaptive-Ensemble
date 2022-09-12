import numpy as np
import random
from copy import deepcopy


def StratifiedScore(y_true, y_score, weight):
    validation = np.ones(len(weight))
    list_positive = []
    list_negative = []
    weight_total = 0
    fitness = 0
    list_labels = list(set(y_true))
    for item in list_labels:
        weight_total += np.exp(-weight[int(np.abs(item) - 1)])
        if item > 0:
            list_positive.append(item)
        else:
            list_negative.append(item)
    ratio_positive = len(list_positive) / (len(list_positive) + len(list_negative))
    ratio_negative = len(list_negative) / (len(list_positive) + len(list_negative))
    constraint = 1 / - (ratio_positive * np.log2(ratio_positive) + ratio_negative * np.log2(ratio_negative))
    for item in list_labels:
        indices = np.where(y_true == item)
        if item in list_positive:
            loss_class = np.mean(np.ones(len(indices)) - y_score[indices][..., 1])
        else:
            loss_class = np.mean(y_score[indices][..., 0])
        validation[int(np.abs(item) - 1)] = 1 - loss_class
        fitness += (1 - constraint * loss_class) * np.exp(-weight[int(np.abs(item) - 1)]) / weight_total
    
    if max(validation) > 1:
        print(validation)
    
    return fitness, validation


class StratifiedUnderSampling(object):
    def __init__(self):
        self.name = 'StratifiedUnderSampling'
    
    def fit_resample(self, X, y):
        self.name = 'StratifiedUnderSampling'
        list_negative = []
        list_positive = []
        list_labels = list(set(y))
        for item in list_labels:
            if item > 0:
                list_positive.append(len(np.where(y == item)[0]))
            else:
                list_negative.append(len(np.where(y == item)[0]))
        
        IR = len(list_negative) / len(list_positive)
        if min(list_negative) >= int(min(list_positive) / IR + 0.5):
            n_positive = min(list_positive)
            n_negative = int(min(list_positive) / IR + 0.5)
        else:
            n_positive = int(min(list_negative) * IR + 0.5)
            n_negative = min(list_negative)
        
        X_return = None
        y_return = None
        for item in list_labels:
            if X_return is None:
                if item > 0:
                    X_return = X[np.where(y == item)][random.sample(range(0, len(np.where(y == item)[0])), n_positive)]
                    y_return = np.ones(n_positive, dtype=int)
                else:
                    X_return = X[np.where(y == item)][random.sample(range(0, len(np.where(y == item)[0])), n_negative)]
                    y_return = np.zeros(n_negative, dtype=int)
            else:
                if item > 0:
                    X_item = X[np.where(y == item)][random.sample(range(0, len(np.where(y == item)[0])), n_positive)]
                    X_return = np.concatenate((X_return, X_item))
                    y_return = np.concatenate((y_return, np.ones(n_positive, dtype=int)))
                else:
                    X_item = X[np.where(y == item)][random.sample(range(0, len(np.where(y == item)[0])), n_negative)]
                    X_return = np.concatenate((X_return, X_item))
                    y_return = np.concatenate((y_return, np.zeros(n_negative, dtype=int)))
        
        return X_return, y_return


class StratifiedUnderBagging(object):
    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = {}
    
    def fit(self, X, y):
        sampler = StratifiedUnderSampling()
        for i in range(0, self.n_estimators):
            X_new, y_new = sampler.fit_resample(X, y)
            model = deepcopy(self.base_estimator)
            model.fit(X_new, y_new)
            self.estimators[i] = deepcopy(model)
    
    def predict(self, X):
        label_pred_proba = self.predict_proba(X)
        label_pred = label_pred_proba[..., 1] > 0.5
        
        return label_pred
    
    def predict_proba(self, X):
        label_pred_proba = np.zeros((len(X), 2))
        for i in range(0, self.n_estimators):
            label_pred_proba += self.estimators[i].predict_proba(X) / self.n_estimators
        
        return label_pred_proba
