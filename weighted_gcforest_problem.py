from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from first_gclayer import FirstGcLayer
from weighted_gcforest import WeightedGcForest


class Problem(object):
    """ Problem class defined the targeted problem

    Note: Need to rewrite fitness() function and subset_size() function if change the problem
    subset_size() is used to calculate the candidate/solution realsize, i.e.,
    actrual selected features or actrual layers amount
    """

    def __init__(self, args):
        self.X_train = args['X_train']
        self.y_train = args['y_train']
        self.X_val = args['X_val']
        self.y_val = args['y_val']
        self.classifiers = args['classifiers']
        self.n_classifiers = len(self.classifiers)
        self.n_classes = args['n_classes']
        self.n_cv_folds = args['n_cv_folds']
        self.concat_type = args['concat_type']
        self.input_type = args['input_type']

        # dimension: max size of a candidate
        self.dimension = args['max_n_layers'] * self.n_classifiers * self.n_classes

        # as we use accuracy as fitness score --> worst fitness = 0.0
        self.worst_fitness = [0.0,0.0]

        self.first_layer = FirstGcLayer(self.classifiers,
                                        self.n_cv_folds,
                                        self.n_classes,
                                        self.n_classifiers,
                                        self.X_train,
                                        self.y_train)

    def fitness(self, candidate, layer_details=False, return_n_layers=False):
        """ Calculate the fitness score for a given candidate

        Parameters
        ----------
        candidate : determine the structure of a GcForest
            candidate = [c_i]
            candidate.shape = n_layers x n_classifiers x n_classes

        layer_details : if True, return error at each layer
        return_n_layers : if True, return config['n_layers']

        Returns
        -------
            accuracy : accuracy of GcForest built from the input candidate
        """

        config = self.candidate_to_config(candidate, self.n_classifiers,
                                          self.n_classes, self.classifiers,
                                          self.n_cv_folds, self.concat_type,
                                          self.input_type)

        if config['n_layers'] == 0:
            return self.worst_fitness

        weighted_gc_forest = WeightedGcForest(config, self.first_layer)
        weighted_gc_forest.fit(self.X_train, self.y_train)

        if layer_details:
            _, layer_errors = weighted_gc_forest.test_details(self.X_val, self.y_val)
            return layer_errors
        else:  # only return accuracy at the last layer
            y_pred = weighted_gc_forest.predict(self.X_val)
            accuracy = accuracy_score(self.y_val, y_pred)
            f1_macro=f1_score(self.y_val-1,y_pred-1,average='macro')

            if return_n_layers:
                return accuracy, config['n_layers']
            else:
                return accuracy,f1_macro

    def compare(self, candidate1, candidate2):
        fitness1, n_layer1 = self.fitness(candidate1)
        fitness2, n_layer2 = self.fitness(candidate2)
        distance = fitness1 - fitness2
        if distance == 0:
            distance = - (n_layer1 - n_layer2)

        return distance

    def get_worst_fitness(self):
        return self.worst_fitness

    def set_dimension(self, dimension):
        self.dimension = dimension

    def get_dimension(self):
        return self.dimension

    def subset_size(self, candidate):
        # Truncate if candidate's length is not divisible by (nClasses x nClassifiers)
        n_layers = int(candidate.shape[0] * 1.0 / (self.n_classes * self.n_classifiers))
        return n_layers * self.n_classes * self.n_classifiers

    @staticmethod
    def candidate_to_config(candidate, n_classifiers, n_classes, classifiers, n_cv_folds, concat_type, input_type):
        """ Convert a PSO candidate to a config determining the structure of GcForest

        Parameters
        ----------
        classifiers : classifier list
        n_classes : number of classes
        n_classifiers : number of classifiers
        n_cv_folds : n_folds used for cross validation
        concat_type : concat with original data or not
        input_type: type of input for next layer
        candidate : PSO candidate
            candidate = [c_i]
            candidate.shape = n_layers x n_classifiers x n_classes

        Returns
        -------
        config :
            config = {
                "n_layers" : L,
                "weights" : [W_0, W_1, ..., W_L]
                "classifiers" : list of models,
                "n_cv_folds" : n_folds for cross validation,
                "n_classes" : number of classes,
                "n_classifiers" : number of classifiers,
                "concat_type": type of concatenation
                "input_type":...
            }

            W_l = [w_ij]
            W_l.shape = n_classifiers x n_classes
        """
        n_layers = int(candidate.shape[0] * 1. / (n_classes * n_classifiers))
        weights = []

        cnt = -1
        for i_layer in range(n_layers):
            W = np.zeros((n_classifiers, n_classes))
            for i_classifier in range(n_classifiers):
                for i_class in range(n_classes):
                    cnt += 1
                    W[i_classifier, i_class] = candidate[cnt]
            weights.append(W)

        config = {
            "n_layers": n_layers,
            "weights": weights,
            "classifiers": classifiers,
            "n_cv_folds": n_cv_folds,
            "n_classes": n_classes,
            "n_classifiers": n_classifiers,
            "concat_type": concat_type,
            "input_type": input_type
        }

        return config
