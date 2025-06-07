from skmultiflow.meta import LeverageBagging, OzaBagging
from skmultiflow.trees import HoeffdingTree
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from ProbabilityCalculation import ProbabilityCalculation

import skmultiflow
print(skmultiflow.__file__)
import numpy as np
import pandas as pd
import argparse
import os

def experiment(real_data, num_classifiers, num_classes, dataset_name, max_samples, iteration):

    if real_data == 0:
        stream = RandomRBFGenerator(n_classes = num_classes, n_features=20 )
    else:
        stream = FileStream(dataset_name)

    stream.prepare_for_use()
    target_values = [ i for i in range(num_classes) ]

    ozaBag = OzaBagging(base_estimator=HoeffdingTree(),
                            n_estimators=num_classifiers)
    

    start_time = pd.Timestamp.now()

    X, y = stream.next_sample(500)
    ozaBag.partial_fit(X, y, classes=target_values)
    print(len(ozaBag.ensemble))

    probabiliy_calculator = ProbabilityCalculation(num_classes, num_classifiers)

    total = 0
    corrects = 0
    while(stream.has_more_samples() and total < max_samples):
        if total % 1000 == 0:
            print("instance", total, flush=True)
        total += 1

        X, y = stream.next_sample()


        proba_votes = np.array([
            proba[0] if proba.shape == (1, num_classes) else np.full(num_classes, 1.0 / num_classes)
            for model in ozaBag.ensemble
            if (proba := model.predict_proba(X)) is not None
        ])    

        probabiliy_calculator.iterateVotes(proba_votes)

        pred = ozaBag.predict(X)
        if pred is not None:
            if y[0] == pred[0]:
                corrects += 1

        ozaBag.partial_fit(X, y)

    end_time = pd.Timestamp.now()
    print("total time: ", end_time - start_time)

    accuracy = corrects / total
    p_array = probabiliy_calculator.p_array
    p_total = probabiliy_calculator.p_total

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