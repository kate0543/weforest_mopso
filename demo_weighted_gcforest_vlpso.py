import time

from mopso.VLPSO import VLPSO
from mopso.local_search import LocalSearch
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from output_writer import OutputWriter
from weighted_gcforest import WeightedGcForest, ConcatenateType, InputType
from weighted_gcforest_problem import Problem
from data_helper import data_folder, file_list

# ------------------------ Parameters ---------------------- #

n_cv_folds = 5
max_n_layers = 5
concat_type = ConcatenateType.CONCAT_WITH_ORIGINAL_DATA
input_type = InputType.SUM_METADATA
n_vlpso_generations = 10
n_vlpso_population = 10
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


for file_name in file_list:
    print(file_name)
    output_writer = OutputWriter('vlpso_result' + '_' + str(n_trees) + '_trees/' + file_name)

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

    # ------------------------ VLPSO ---------------------------- #
    '''parameters for vlpso class'''
    dimension = n_classifiers * n_classes * max_n_layers
    chunk_size = n_classifiers * n_classes
    para1 = {'dimension': dimension, 'chunk_size': chunk_size, 'layers': max_n_layers, 'iteration': n_vlpso_generations,
             'population': n_vlpso_population, 'LOCAL_SEARCH': True, 'REINIT': True, 'LS_MAX_ITER': 100}
    '''parameters for local search class'''
    para2 = {'percentage': 0.25, 'max_iter': 100}
    ''' create a local search object'''
    localsearch = LocalSearch(para2, problem)
    '''create a varieble length pso object'''
    vlpso = VLPSO(para1, problem, localsearch)
    '''perform search'''
    best_candidate, best_fitness, generation_output, best_output = vlpso.VELPSO()

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
    vlpso_output = {
        "generations_average_fitness": generation_output,
        "generations_best_fitness": best_output
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
        "n_vlpso_generations": n_vlpso_generations,
        "n_vlpso_population": n_vlpso_population,
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
    output_writer.write_output(vlpso_output, 'vlpso_output', indent=2)
    output_writer.write_output(final_output, 'performance', indent=2)
    output_writer.write_output(parameters, 'parameters', indent=2)
    output_writer.write_output(layer_errors, 'layer_errors', indent=2)
    output_writer.write_output(time_output, 'runtime', indent=2)
