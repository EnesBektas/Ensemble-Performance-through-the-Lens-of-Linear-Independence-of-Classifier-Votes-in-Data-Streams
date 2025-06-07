from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator

import numpy as np
from Goowe import Goowe
import pandas as pd
import time
from scipy.io import arff
import argparse
import os


def experiment(real_data, num_classifiers, num_classes, dataset_name, max_samples, iteration):

    if real_data == 0:
        stream = RandomRBFGenerator(n_classes = num_classes, n_features=20 )
    else:
        stream = FileStream(dataset_name)


    stream.prepare_for_use()

    num_features = stream.n_features
    num_targets = 1 #stream.n_targets

    target_values = [ i for i in range(num_classes) ]

    N_MAX_CLASSIFIERS = num_classifiers
    CHUNK_SIZE = 500        # User-specified
    WINDOW_SIZE = 100       # User-specified

    # Initialize the ensemble
    goowe = Goowe(n_max_components=N_MAX_CLASSIFIERS,
                chunk_size=CHUNK_SIZE,
                window_size=WINDOW_SIZE,
                logging = False)
    goowe.prepare_post_analysis_req(num_features, num_targets, num_classes, target_values)

    # For the first chunk, there is no prediction.
    X_init, y_init = stream.next_sample(CHUNK_SIZE)

    goowe.partial_fit(X_init, y_init)

    accuracy = 0.0
    total = 0.0
    true_predictions = 0.0

    for i in range(CHUNK_SIZE):
        total += 1
        cur = stream.next_sample()
        X, y = cur[0], cur[1]
        #  print(X)
        preds = goowe.predict(X)
        
        true_predictions += np.sum(preds == y)
        accuracy = true_predictions / total
        #  print('\tData instance: {} - Accuracy: {}'.format(total, accuracy))
        
        goowe.partial_fit(X, y)

    k = 0
    while(stream.has_more_samples() and k < max_samples):
        k += 1
        total += 1
        
        cur = stream.next_sample()
        X, y = cur[0], cur[1]
        
        preds = goowe.predict(X)            # Test
        true_predictions += np.sum(preds == y)

        goowe.partial_fit(X, y)             # Then train
        

    accuracy = true_predictions / total

    p_array = goowe.probability_calculator.p_array
    p_total = goowe.probability_calculator.p_total
    print("accuracy: ", accuracy)   
    print("p_array: ", p_array)
    print("p_total: ", p_total)

    print("probabilities: ", p_array/p_total)

    if real_data == 0:
        folder_name = "RBF" + str(num_classes) + "classes/" + str(num_classifiers) + "classifiers/iteration" + str(iteration)
    else:
        folder_name = os.path.splitext(dataset_name)[0] + "/" + str(num_classifiers) + "classifiers/iteration" + str(iteration)

    try:
        os.makedirs(folder_name)
    except:
        pass

    np.save(folder_name + "/p_array", p_array)
    np.save(folder_name + "/p_total", p_total)
    np.save(folder_name + "/accuracy", accuracy)


parser = argparse.ArgumentParser()

parser.add_argument('--real_data', type=int, default=0 )
parser.add_argument('--dataset', type=str, default='elec.csv' )
parser.add_argument('--classifiers', type=int, default=2 )
parser.add_argument('--num_classes', type=int, default=2 )
parser.add_argument('--max_samples', type=int, default=1000000 )
parser.add_argument('--num_of_iterations', choices=[i for i in range(1, 11)], type=int, default=1)
parser.add_argument('--start_index_of_iteration', type=int, default=0)

args = parser.parse_args()

real_data = args.real_data
num_classifiers = args.classifiers
num_classes = args.num_classes
dataset_name = args.dataset
max_samples = args.max_samples
num_of_iterations = args.num_of_iterations
start_index_of_iteration = args.start_index_of_iteration

for i in range(start_index_of_iteration, start_index_of_iteration + num_of_iterations):
    experiment(real_data, num_classifiers, num_classes, dataset_name, max_samples, i)
