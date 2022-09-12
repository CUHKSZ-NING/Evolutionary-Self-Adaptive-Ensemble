import random
import math
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from Tools import StratifiedUnderBagging, StratifiedScore


class Individual:
    def __init__(self, num_base_estimators=0, dynamic_classifier=True, dynamic_feature=True, dynamic_hierarchical=True,
                 randomness=0.2):
        self.fitness = 0
        self.accuracy = 0
        self.num_base_estimators_ = num_base_estimators
        self.dynamic_classifier_ = dynamic_classifier
        self.dynamic_feature_ = dynamic_feature
        self.dynamic_hierarchical_ = dynamic_hierarchical
        self.randomness_ = randomness
        self.validation = None
        
        self.chromosome = {
            'feature': np.zeros(0),
            'label': np.zeros(0),
            'clf': 0
        }
        
        self.cor = {
            'feature': 0,
            'label': 0,
            'clf': 0
        }
        self.mr = {
            'feature': 0,
            'label': 0,
            'clf': 0
        }
    
    def initialize(self, feature_dim, node, label_next, features_dropped):
        alpha = 0.9
        feature_probability = alpha + (1 - alpha) * random.random()
        
        while True:
            if self.dynamic_feature_ is False:
                self.chromosome['feature'] = np.ones(feature_dim, dtype=int)
                break
            
            self.chromosome['feature'] = np.zeros(feature_dim, dtype=int)
            for i in range(0, feature_dim):
                if features_dropped[i]:
                    self.chromosome['feature'][i] = -1
                elif random.random() < feature_probability:
                    self.chromosome['feature'][i] = 1
            
            if 1 in self.chromosome['feature']:
                break
        
        while True:
            self.chromosome['label'] = np.zeros(len(node), dtype=int)
            for i in range(0, len(node)):
                if node[i] == label_next:
                    self.chromosome['label'][i] = random.randint(0, 1)
                else:
                    self.chromosome['label'][i] = -1
            
            if 0 in self.chromosome['label'] and 1 in self.chromosome['label']:
                break
        
        self.chromosome['clf'] = random.randint(0, self.num_base_estimators_ - 1)
    
    def get_fitness(self, X, y, base_estimators, weight):
        if self.dynamic_classifier_ is False:
            self.chromosome['clf'] = random.randint(0, self.num_base_estimators_ - 1)
        
        self.fitness = 0
        
        name_base_estimators = {}
        index = 0
        
        for key in base_estimators:
            name_base_estimators[index] = key
            index += 1
        
        class_count = np.zeros(2, dtype=int)
        
        feature_index = []
        
        for i in range(0, len(self.chromosome['feature'])):
            if self.chromosome['feature'][i] == 1:
                feature_index.append(i)
        
        data_temp = X[..., feature_index]
        
        data_dec = {
            0: {},
            1: {}
        }

        for i in range(0, len(self.chromosome['label'])):
            if self.chromosome['label'][i] != -1:
                class_count[self.chromosome['label'][i]] += 1
                data_dec[self.chromosome['label'][i]][i] = data_temp[np.where(y == i)]
        
        if 0 in class_count or 1 not in self.chromosome['feature']:  # ?
            self.fitness = 0
            
            return self.fitness
        
        X_new = None
        y_new = None
        
        for label in data_dec:
            for key in data_dec[label]:
                if X_new is None:
                    X_new = data_dec[label][key]
                    y_new = np.ones(len(data_dec[label][key]), dtype=int) * int(key + 1) * int(label * 2 - 1)
                else:
                    X_new = np.concatenate((X_new, data_dec[label][key]))
                    y_new = np.concatenate(
                        (y_new, np.ones(len(data_dec[label][key]), dtype=int) * int(key + 1) * int(label * 2 - 1)))
        
        n_folds = 2
        self.fitness = 0
        self.accuracy = 0
        self.validation = None
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        for index_train, index_test in skf.split(X_new, y_new):
            X_train = X_new[index_train]
            y_train = y_new[index_train]
            X_test = X_new[index_test]
            y_test = y_new[index_test]
            model = deepcopy(base_estimators[name_base_estimators[self.chromosome['clf']]])
            model = StratifiedUnderBagging(base_estimator=model, n_estimators=1)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)
            
            score, validation = StratifiedScore(y_test, y_pred_proba[..., 1], weight)
            
            if self.validation is None:
                self.validation = validation / n_folds
            else:
                self.validation += validation / n_folds
            self.fitness += score / n_folds
            self.accuracy += balanced_accuracy_score(y_test, y_pred_proba[..., 1] >= 0.5) / n_folds
    
    def cross_over(self, other):
        individual = Individual(num_base_estimators=self.num_base_estimators_)
        individual.chromosome = deepcopy(self.chromosome)
        
        for i in range(0, len(self.chromosome['label'])):
            if random.random() < self.cor['label']:
                individual.chromosome['label'][i] = other.chromosome['label'][i]
        
        for i in range(0, len(self.chromosome['feature'])):
            if random.random() < self.cor['feature']:
                individual.chromosome['feature'][i] = other.chromosome['feature'][i]
        
        if random.random() < self.cor['clf']:
            individual.chromosome['clf'] = other.chromosome['clf']
        
        return individual
    
    def mutate(self):
        for i in range(0, len(self.chromosome['label'])):
            if self.chromosome['label'][i] == -1:
                continue
            
            if random.random() < self.mr['label']:
                if self.chromosome['label'][i] == 0:
                    self.chromosome['label'][i] = 1
                else:
                    self.chromosome['label'][i] = 0
        
        for i in range(0, len(self.chromosome['feature'])):
            if self.chromosome['feature'][i] == -1:
                continue
            
            if random.random() < self.mr['feature']:
                if self.chromosome['feature'][i] == 0:
                    self.chromosome['feature'][i] = 1
                else:
                    self.chromosome['feature'][i] = 0
        
        if random.random() < self.mr['clf']:
            self.chromosome['clf'] = random.randint(0, self.num_base_estimators_ - 1)
        
        return
    
    def rate_update(self, label_count, progress):
        flexibility = (label_count - 2) / label_count
        
        self.cor = {
            'feature': self.randomness_,
            'label': self.randomness_ * flexibility,
            'clf': self.randomness_ * math.exp(-progress)
        }
        self.mr = {
            'feature': 0.1 * self.randomness_ * math.exp(progress),
            'label': 0.1 * self.randomness_ * flexibility * math.exp(-progress),
            'clf': 0.1 * self.randomness_ * math.exp(-progress)
        }
        
        if self.dynamic_hierarchical_ is False:
            self.cor['label'] = 0
            self.mr['label'] = 0
        
        if self.dynamic_feature_ is False:
            self.cor['feature'] = 0
            self.mr['feature'] = 0
        
        return
