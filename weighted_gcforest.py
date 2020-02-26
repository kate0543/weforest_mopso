from sklearn.metrics import accuracy_score

from gclayer import GcLayer
import numpy as np
from enum import Enum


class ConcatenateType(str, Enum):
    METADATA_ONLY = "metadata_only"
    CONCAT_WITH_ORIGINAL_DATA = "concat_with_original_data"


class InputType(str, Enum):
    SUM_METADATA = "sum_metadata"
    METADATA = "metadata"


class WeightedGcForest:

    def __init__(self, config, first_layer=None):
        self.n_layers = config['n_layers']
        self.weights = config['weights']
        self.classifiers = config['classifiers']
        self.n_cv_folds = config['n_cv_folds']
        self.n_classes = config['n_classes']
        self.n_classifiers = config['n_classifiers']
        self.concat_type = config['concat_type']
        self.input_type = config['input_type']

        if first_layer is None:
            self.layers = []
        else:
            self.layers = [first_layer]

    def fit(self, X, y):
        XL = np.array(X)
        yL = np.array(y)
        for i_layer in range(self.n_layers):

            # if caching first layer, use it so that
            # classifiers at layer 1 don't need to retrain
            if i_layer == 0 and len(self.layers) > 0:
                layer = self.layers[i_layer]
                layer.set_weight(self.weights[i_layer])
                weighted_sum_metadata, weighted_metadata = layer.cross_validation(XL, yL)
            else:
                layer = GcLayer(self.classifiers,
                                self.n_cv_folds,
                                self.n_classes,
                                self.n_classifiers,
                                weight=self.weights[i_layer])
                weighted_sum_metadata, weighted_metadata = layer.cross_validation(XL, yL)
                layer.fit(XL, yL)
                self.layers.append(layer)

            XL = self.get_next_layer_input(weighted_metadata, weighted_sum_metadata, X)

    def predict_proba(self, X):
        XL = np.array(X)
        weighted_sum_metadata = None
        for i_layer in range(self.n_layers):
            weighted_sum_metadata, weighted_metadata = self.layers[i_layer].predict_proba(XL)
            XL = self.get_next_layer_input(weighted_metadata, weighted_sum_metadata, X)

        return weighted_sum_metadata

    def predict(self, X):
        weighted_sum_metadata = self.predict_proba(X)
        prediction = np.argmax(weighted_sum_metadata, axis=1) + 1
        return prediction

    def get_next_layer_input(self, weighted_metadata, weighted_sum_metadata, X):
        """
        """

        if self.input_type == InputType.METADATA:
            input = weighted_metadata
        elif self.input_type == InputType.SUM_METADATA:
            input = weighted_sum_metadata
        else:
            raise Exception("Unimplemented exception")

        if self.concat_type == ConcatenateType.CONCAT_WITH_ORIGINAL_DATA:
            next_layer_input = np.concatenate((input, X), axis=1)
        elif self.concat_type == ConcatenateType.METADATA_ONLY:
            next_layer_input = np.array(input)
        else:
            raise Exception("Unimplemented exception")

        return next_layer_input

    def test_details(self, X, y):
        """ Use each layer to predict X, return accuracy by comparing to y
        """
        XL = np.array(X)
        layer_errors = []
        weighted_sum_metadata = None
        for i_layer in range(self.n_layers):
            weighted_sum_metadata, weighted_metadata = self.layers[i_layer].predict_proba(XL)
            prediction = np.argmax(weighted_sum_metadata, axis=1) + 1
            accuracy = accuracy_score(y, prediction)
            layer_errors.append(1.0 - accuracy)

            XL = self.get_next_layer_input(weighted_metadata, weighted_sum_metadata, X)

        return weighted_sum_metadata, layer_errors
