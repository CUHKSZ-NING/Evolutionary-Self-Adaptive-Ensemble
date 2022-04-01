import random
import math
import numpy as np
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.ensemble import BalancedBaggingClassifier as BBC

# from Tools import random_split
# from Tools import cross_split
# from Tools import get_fold
# from ComparativeTrials import get_model


class Individual:
    def __init__(self, num_base_estimators=0, dynamic_classifier=True, dynamic_feature=True, dynamic_hierarchical=True,
                 gamma=0.02):
        self.fitness = 0
        self.accuracy = 0
        self.num_base_estimators_ = num_base_estimators
        self.dynamic_classifier_ = dynamic_classifier
        self.dynamic_feature_ = dynamic_feature
        self.dynamic_hierarchical_ = dynamic_hierarchical
        self.gamma_ = gamma
        # print(self.num_base_estimators_)

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

    def initialize(self, feature_dim, node, label_next, features_droped):
        lbd = 0.5
        feature_probability = lbd + (1 - lbd) * random.random()

        while True:
            if self.dynamic_feature_ is False:
                self.chromosome['feature'] = np.ones(feature_dim, dtype=int)
                break

            self.chromosome['feature'] = np.zeros(feature_dim, dtype=int)
            for i in range(0, feature_dim):
                if features_droped[i]:
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

    def get_fitness(self, X, y, base_estimators, function='default'):
        if self.dynamic_classifier_ is False:
            self.chromosome['clf'] = random.randint(0, self.num_base_estimators_ - 1)

        self.fitness = 0

        name_base_estimators = {}
        index = 0
        for key in base_estimators:
            name_base_estimators[index] = key
            index += 1

        class_count = np.zeros(2, dtype=int)
        data_temp = deepcopy(X)

        feature_index = []

        for i in range(0, len(self.chromosome['feature'])):
            if self.chromosome['feature'][i] == 1:
                feature_index.append(i)

        data_temp = data_temp[..., feature_index]

        data_dec = {
            0: {},
            1: {}
        }

        for i in range(0, len(self.chromosome['label'])):
            if self.chromosome['label'][i] != -1:
                class_count[self.chromosome['label'][i]] += 1
                data_dec[self.chromosome['label'][i]][i] = deepcopy(data_temp[np.where(y == i)])

        if 0 in class_count or 1 not in self.chromosome['feature']:  # ?
            self.fitness = 0

            return self.fitness

        X_new = None
        y_new = None

        for label in data_dec:
            for key in data_dec[label]:
                if X_new is None:
                    X_new = deepcopy(data_dec[label][key])
                    y_new = np.ones(len(data_dec[label][key]), dtype=int) * int(label)
                else:
                    X_new = np.concatenate((X_new, data_dec[label][key]))
                    y_new = np.concatenate((y_new, np.ones(len(data_dec[label][key]), dtype=int) * int(label)))

        cmatrix = None

        skf = StratifiedKFold(n_splits=2, shuffle=True)
        for index_train, index_test in skf.split(X_new, y_new):
            X_train = deepcopy(X_new[index_train])
            y_train = deepcopy(y_new[index_train])
            X_test = deepcopy(X_new[index_test])
            y_test = deepcopy(y_new[index_test])
            IR = max(len(data_dec[0]), len(data_dec[1])) / min(len(data_dec[0]), len(data_dec[1]))
            model = BBC(base_estimator=deepcopy(base_estimators[name_base_estimators[self.chromosome['clf']]]),
                        n_estimators=int(IR + 0.5))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            matrix_i = confusion_matrix(y_test, y_pred)
            if cmatrix is None:
                cmatrix = deepcopy(matrix_i)
            else:
                cmatrix += deepcopy(matrix_i)

        if function in ['f1', 'IoU']:
            if function == 'F1':
                score = 0
                precision = np.array([cmatrix[0, 0] / sum(cmatrix[..., 0]), cmatrix[1, 1] / sum(cmatrix[..., 1])])
                recall = np.array([cmatrix[0, 0] / sum(cmatrix[0, ...]), cmatrix[1, 1] / sum(cmatrix[1, ...])])

                for i in [0, 1]:
                    score += precision[i] * recall[i] / (precision[i] + recall[i])

            else:
                w = np.zeros(2)
                w[0] = cmatrix[0, 0] / (cmatrix[0, 0] + cmatrix[0, 1] + cmatrix[1, 0])
                w[1] = cmatrix[1, 1] / (cmatrix[1, 1] + cmatrix[1, 0] + cmatrix[0, 1])

                score = 0.5 * sum(w)

            self.fitness = score

        else:
            score = np.array([0.5 * (cmatrix[0, 0] / sum(cmatrix[0, ...]) + cmatrix[0, 0] / sum(cmatrix[..., 0])),
                              0.5 * (cmatrix[1, 1] / sum(cmatrix[1, ...]) + cmatrix[1, 1] / sum(cmatrix[..., 1]))])

            weight = [sum(class_count) / class_count[0], sum(class_count) / class_count[1]]
            step = [score[0] * math.log(weight[0], 2) / weight[0], score[1] * math.log(weight[1], 2) / weight[1]]
            self.fitness = max(0.01, 1 - (1 - sum(score) / 2) / sum(step))
            self.accuracy = score[0] / weight[0] + score[1] / weight[1]

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
            'feature': self.gamma_,
            'label': self.gamma_ * flexibility,
            'clf': self.gamma_ * math.exp(-progress)
        }
        self.mr = {
            'feature': 0.1 * self.gamma_ * math.exp(progress),
            # 'feature': 0,
            'label': 0.1 * self.gamma_ * flexibility * math.exp(-progress),
            'clf': 0.1 * self.gamma_ * math.exp(-progress)
        }

        if self.dynamic_hierarchical_ is False:
            self.cor['label'] = 0
            self.mr['label'] = 0

        if self.dynamic_feature_ is False:
            self.cor['feature'] = 0
            self.mr['feature'] = 0

        return
