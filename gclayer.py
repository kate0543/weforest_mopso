import copy
from sklearn.model_selection import KFold
import numpy as np


class GcLayer:
    def __init__(self, classifiers, n_cv_folds, n_classes, n_classifiers, weight=None):
        """
        Parameters
        ----------
        weight : [w_ij], weight.shape = n_classifiers x n_classes
        """

        self.weight = weight
        self.classifiers = classifiers
        self.trained_models = []
        self.n_cv_folds = n_cv_folds
        self.n_classes = n_classes
        self.n_classifiers = n_classifiers

    def fit(self, X, y):
        """ learn all classifiers using X, y
            --> trained models used to predict test set
        """

        for classifier in self.classifiers:
            new_classifier = copy.deepcopy(classifier)
            new_classifier.fit(X, y)
            self.trained_models.append(new_classifier)

    def cross_validation(self, X, y):
        """ Apply K-folds cross validation on (X, y)
            to obtain the metadata

        Parameters
        ----------
        X : features

        y : label

        Returns
        -------
        weighted_sum_metadata : n_instances x n_classes
        """

        n_instances = X.shape[0]
        weighted_sum_metadata = np.zeros((n_instances, self.n_classes))
        weighted_metadata = np.zeros((n_instances, self.n_classes * self.n_classifiers))

        kf = KFold(n_splits=self.n_cv_folds)
        kf_split = kf.split(X)

        for train_ids, test_ids in kf_split:
            X_train = X[train_ids, :]
            y_train = y[train_ids]

            X_test = X[test_ids, :]

            weighted_sum_prob = np.zeros((X_test.shape[0], self.n_classes))
            weighted_prob = np.zeros((X_test.shape[0], 0))

            for i_classifier in range(self.n_classifiers):
                new_classifier = copy.deepcopy(self.classifiers[i_classifier])
                new_classifier.fit(X_train, y_train)
                prob = new_classifier.predict_proba(X_test)

                w = self.weight[i_classifier, :]

                #   prob.shape = (n_test, n_classes)
                #   w.shape = (1, n_classes)
                #   np.multiply(prob, w) --> broadcast
                weighted_sum_prob = weighted_sum_prob + np.multiply(prob, w)
                weighted_prob = np.concatenate((weighted_prob, np.multiply(prob, w)), axis=1)

            weighted_sum_metadata[test_ids, :] = weighted_sum_prob
            weighted_metadata[test_ids, :] = weighted_prob

        return weighted_sum_metadata, weighted_metadata

    def predict(self, X):
        """ use trained_models to predict for X

        Parameters
        ---------
        X :

        Returns
        -------
        """
        raise Exception("Unimplemented exception!")

    def predict_proba(self, X):
        """ use trained_models to predict probabilities for X

        Parameters
        ----------
        X : unlabeled data (n_instances x n_features)

        Returns
        -------
        weighted_sum_metadata : (n_instances x n_classes)
        """
        n_instances = X.shape[0]
        weighted_sum_metadata = np.zeros((n_instances, self.n_classes))
        weighted_metadata = np.zeros((n_instances, 0))

        for i_classifier in range(self.n_classifiers):
            prob = self.trained_models[i_classifier].predict_proba(X)
            w = self.weight[i_classifier, :]

            weighted_sum_metadata = weighted_sum_metadata + np.multiply(prob, w)
            weighted_metadata = np.concatenate((weighted_metadata, np.multiply(prob, w)), axis=1)

        return weighted_sum_metadata, weighted_metadata
