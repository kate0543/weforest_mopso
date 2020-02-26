import sys
import time
import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from data_helper import data_folder, file_list
from output_writer import OutputWriter
from vlga.vlga import VLGA, SelectionType
from weighted_gcforest import WeightedGcForest, ConcatenateType, InputType
from weighted_gcforest_problem import Problem

NAME = "VLGA"

try:
    # run with file_list[from_id, from_id + 1,..., to_id - 1]
    from_id = int(sys.argv[1])
    to_id = int(sys.argv[2])
except:
    from_id = 0
    to_id = len(file_list)

# ------------------------ Parameters ---------------------- #

n_cv_folds = 5
max_n_layers = 5
concat_type = ConcatenateType.CONCAT_WITH_ORIGINAL_DATA
input_type = InputType.SUM_METADATA
n_vlga_generations = 5
n_vlga_population = 5
vlga_selection_type = SelectionType.ROULETTE
n_trees = 10


# ------------------------ Parameters ---------------------- #

def init_classifiers():
    prf1_str = "RandomForestClassifier(n_estimators=n_trees, max_features='sqrt', random_state=1, " \
               "min_samples_split=0.1, n_jobs=1)"

    prf2_str = "RandomForestClassifier(n_estimators=n_trees, max_features='sqrt', random_state=2, " \
               "min_samples_split=0.1, n_jobs=1)"

    crf1_str = "RandomForestClassifier(n_estimators=n_trees, max_features=1, random_state=1, " \
               "min_samples_split=0.1, n_jobs=1)"

    crf2_str = "RandomForestClassifier(n_estimators=n_trees, max_features=1, random_state=2, " \
               "min_samples_split=0.1, n_jobs=1)"

    prf1 = eval(prf1_str)
    prf2 = eval(prf2_str)
    crf1 = eval(crf1_str)
    crf2 = eval(crf2_str)

    return [prf1, prf2, crf1, crf2], [prf1_str, prf2_str, crf1_str, crf2_str]


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

    classifiers, classifiers_str = init_classifiers()
    n_classifiers = len(classifiers)

    # ------------ Define Optimization Problem ----------------- #
    train_time_start = time.time()
    problem_args = {
        "X_train": X_train,
        "y_train": Y_train,
        "X_val": X_val,
        "y_val": Y_val,
        "classifiers": classifiers,
        "n_classes": n_classes,
        "n_cv_folds": n_cv_folds,
        "max_n_layers": max_n_layers,
        "concat_type": concat_type,
        "input_type": input_type
    }

    problem = Problem(problem_args)

    # ------------------------ VLGA ---------------------------- #
    vlga = VLGA(
        chromosome_chunk_size=n_classifiers * n_classes,
        max_n_chunk=max_n_layers,
        n_generations=n_vlga_generations,
        n_population=n_vlga_population,
        selection_type=vlga_selection_type,
        random_state=1
    )
    best_candidate, best_fitness = vlga.solve(problem)

    # ----------------- Retrain phase -------------------------- #
    config = Problem.candidate_to_config(best_candidate, n_classifiers, n_classes,
                                         classifiers, n_cv_folds, concat_type, input_type)

    wgf = WeightedGcForest(config)

    X_train_full = np.concatenate((X_train, X_val), axis=0)
    Y_train_full = np.concatenate((Y_train, Y_val))

    wgf.fit(X_train_full, Y_train_full)  # retrain model

    train_time_end = time.time()

    # Get more information for paper report -> shouldn't count time of this step
    val_layer_errors = problem.fitness(best_candidate, layer_details=True)
    print('val layer_errors =', val_layer_errors)

    # ------------------ Testing phase ------------------------- #

    # Get more information for paper report -> shouldn't count time of this step
    _, test_layer_errors = wgf.test_details(X_test, Y_test)
    print('test layer_errors =', test_layer_errors)

    test_time_start = time.time()
    prediction = wgf.predict(X_test)

    # --------------------------------------------------------- #
    error = 1 - accuracy_score(Y_test, prediction)
    print('error =', error)
    micro_f1 = f1_score(Y_test - 1, prediction - 1, average='micro')
    print('micro_f1 =', micro_f1)
    macro_f1 = f1_score(Y_test - 1, prediction - 1, average='macro')
    print('macro_f1 =', macro_f1)

    test_time_end = time.time()

    # ----------------- Writing Output ------------------------- #

    best_candidate_output = {
        "fitness": best_fitness,
        "n_layers": config["n_layers"],
        "n_classes": n_classes,
        "n_classifiesr": n_classifiers,
        "weights": [weight.tolist() for weight in config["weights"]],
    }
    vlga_output = {
        "generations_average_fitness": vlga.get_result_collector().get_generations_average_fitness(),
        "generations_best_fitness": vlga.get_result_collector().get_generations_best_fitness()
    }

    final_output = {
        "error": error,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

    parameters = {
        "n_cv_folds": n_cv_folds,
        "max_n_layers": max_n_layers,
        "concat_type": concat_type,
        "input_type": input_type,
        "n_vlga_generations": n_vlga_generations,
        "n_vlga_population": n_vlga_population,
        "vlga_selection_type": vlga_selection_type,
        "n_trees": n_trees,
        "classifiers": classifiers_str
    }

    layer_errors = {
        "test_layer_errors": test_layer_errors,
        "val_layer_errors": val_layer_errors
    }

    time_output = {
        "train_time": train_time_end - train_time_start,
        "test_time": test_time_end - test_time_start
    }

    output_writer.write_output(best_candidate_output, 'best_candidate')
    output_writer.write_output(vlga_output, 'vlga_output', indent=2)
    output_writer.write_output(final_output, 'performance', indent=2)
    output_writer.write_output(parameters, 'parameters', indent=2)
    output_writer.write_output(layer_errors, 'layer_errors', indent=2)
    output_writer.write_output(time_output, 'runtime', indent=2)
