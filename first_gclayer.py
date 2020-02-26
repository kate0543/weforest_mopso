import copy

from sklearn.model_selection import KFold

from gclayer import GcLayer
import numpy as np


class FirstGcLayer(GcLayer):
    def __init__(self, classifiers, n_cv_folds, n_classes, n_classifiers, X, y):
        super().__init__(classifiers, n_cv_folds, n_classes, n_classifiers)
        super().fit(X, y)

        self.n_instances = X.shape[0]
        self.cv_prob = [[] for _ in range(n_classifiers)]

    def set_weight(self, weight):
        self.weight = weight

    def fit(self, X, y):
        print('Already fit when calling constructor')
        pass

    def cross_validation(self, X, y):
        self.check_weight()

        weighted_sum_metadata = np.zeros((self.n_instances, self.n_classes))
        weighted_metadata = np.zeros((self.n_instances, self.n_classes * self.n_classifiers))

        # The cross-validation is deterministic
        # because the default value of shuffle=False
        # in KFold()
        kf = KFold(n_splits=self.n_cv_folds)
        kf_split = kf.split(X)

        cnt_fold = -1
        for train_ids, test_ids in kf_split:
            cnt_fold += 1
            X_train = X[train_ids, :]
            y_train = y[train_ids]

            X_test = X[test_ids, :]

            weighted_sum_prob = np.zeros((X_test.shape[0], self.n_classes))
            weighted_prob = np.zeros((X_test.shape[0], 0))

            for i_classifier in range(self.n_classifiers):

                # If this is not the first time -> summon the saved predictions
                if len(self.cv_prob[i_classifier]) > cnt_fold:
                    prob = self.cv_prob[i_classifier][cnt_fold]
                else:  # If this is the first time -> fit and predict
                    new_classifier = copy.deepcopy(self.classifiers[i_classifier])
                    new_classifier.fit(X_train, y_train)
                    prob = new_classifier.predict_proba(X_test)
                    self.cv_prob[i_classifier].append(prob)

                w = self.weight[i_classifier, :]

                #   prob.shape = (n_test, n_classes)
                #   w.shape = (1, n_classes)
                #   np.multiply(prob, w) --> broadcast
                weighted_sum_prob = weighted_sum_prob + np.multiply(prob, w)
                weighted_prob = np.concatenate((weighted_prob, np.multiply(prob, w)), axis=1)

            weighted_sum_metadata[test_ids, :] = weighted_sum_prob
            weighted_metadata[test_ids, :] = weighted_prob

        return weighted_sum_metadata, weighted_metadata

    def check_weight(self):
        if self.weight is None:
            raise Exception("Must set weight for first layer before using")
