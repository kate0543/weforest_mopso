import datetime
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from data_helper import data_folder, file_list
from output_writer import OutputWriter

NAME = "RandomForest"

try:
    # run with file_list[from_id, from_id + 1,..., to_id - 1]
    from_id = int(sys.argv[1])
    to_id = int(sys.argv[2])
except:
    from_id = 0
    to_id = len(file_list)

# ------------------------ Parameters ---------------------- #

n_trees = 500


# ------------------------ Parameters ---------------------- #

def init_classifier():
    rf_str = "RandomForestClassifier(n_estimators=n_trees, max_features='sqrt', random_state=1, " \
             "min_samples_split=0.1, n_jobs=1)"

    rf = eval(rf_str)

    return rf, rf_str


for i_file in range(from_id, to_id):
    file_name = file_list[i_file]
    print(datetime.datetime.now(), ' File {}: '.format(i_file), file_name)
    output_writer = OutputWriter('result/{}/'.format(NAME) + file_name)

    # ---------------------- Prepare Data ---------------------- #
    D_train = np.loadtxt(data_folder + 'train1/' + file_name + '_train1.dat', delimiter=',')
    D_val = np.loadtxt(data_folder + 'val/' + file_name + '_val.dat', delimiter=',')
    D_test = np.loadtxt(data_folder + 'test/' + file_name + '_test.dat', delimiter=',')

    X_train = D_train[:, :-1]
    Y_train = D_train[:, -1].astype(np.int32)
    X_val = D_val[:, :-1]
    Y_val = D_val[:, -1].astype(np.int32)
    X_test = D_test[:, :-1]
    Y_test = D_test[:, -1].astype(np.int32)

    classes = np.unique(np.concatenate((Y_train, Y_val, Y_test)))
    if np.any(classes.astype(np.int32) == 0):
        raise Exception("Labels have to start from 1")

    n_classes = np.size(classes)

    classifier, classifier_str = init_classifier()

    # ----------------- Training phase ------------------------- #
    train_time_start = time.time()

    X_train_full = np.concatenate((X_train, X_val), axis=0)
    Y_train_full = np.concatenate((Y_train, Y_val))

    classifier.fit(X_train_full, Y_train_full)  # retrain model

    train_time_end = time.time()

    # ------------------ Testing phase ------------------------- #
    test_time_start = time.time()

    prediction = classifier.predict(X_test)

    # --------------------------------------------------------- #
    error = 1 - accuracy_score(Y_test, prediction)
    print('error =', error)
    micro_f1 = f1_score(Y_test - 1, prediction - 1, average='micro')
    print('micro_f1 =', micro_f1)
    macro_f1 = f1_score(Y_test - 1, prediction - 1, average='macro')
    print('macro_f1 =', macro_f1)

    test_time_end = time.time()

    # ----------------- Writing Output ------------------------- #

    final_output = {
        "error": error,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

    parameters = {
        "n_trees": n_trees,
        "classifiers": classifier_str
    }

    time_output = {
        "train_time": train_time_end - train_time_start,
        "test_time": test_time_end - test_time_start
    }

    output_writer.write_output(final_output, 'performance', indent=2)
    output_writer.write_output(parameters, 'parameters', indent=2)
    output_writer.write_output(time_output, 'runtime', indent=2)
