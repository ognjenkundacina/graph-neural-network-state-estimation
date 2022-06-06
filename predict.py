import ignnition
import matplotlib.pyplot as plt
from csv import reader
import os
import numpy as np
from statistics import mean
import time
import tensorflow as tf


def predict():
    model = ignnition.create_model(model_dir='./')

    start = time.time()
    predictions = model.predict()
    end = time.time()
    print("Time elapsed: ", end - start)

    predictions = np.array(predictions)
    print(predictions[:, :, 0])

    #zbog implementacije smo u labele morali dodati i iGraph, pa zato i njega imamo u predikcijama, a potreno je uzeti samo napone
    only_voltage_vector = predictions[:, :, 0]

    calculate_test_set_results(only_voltage_vector)

    example_id = 0
    plot_results_for_one_test_example(only_voltage_vector, example_id)


def calculate_test_set_results(predictions_whole_test_set):
    data_dir=os.path.abspath("./data_from_wls_se_solver")
    path=str(data_dir) + "/Test_Estimate.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        estimate_rows = list(csv_reader)

    if len(estimate_rows) != len(predictions_whole_test_set):
        print("ERROR: len(estimate_rows) != len(predictions)")

    MSEs = []
    for i in range(len(estimate_rows)):
        true_values = [float(x) for x in estimate_rows[i]]
        predictions = [float(x) for x in predictions_whole_test_set[i]]
        difference_array = np. subtract(true_values, predictions)
        squared_array = np. square(difference_array)
        mse = squared_array. mean()
        MSEs.append(mse)

    print("Average MSE over the test set: ", mean(MSEs))


def plot_results_for_one_test_example(predictions_whole_test_set, example_id):
    data_dir=os.path.abspath("./data_from_wls_se_solver")
    path=str(data_dir) + "/Test_Estimate.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        estimate_rows = list(csv_reader)

    true_values = [float(x) for x in estimate_rows[example_id]]
    predictions = predictions_whole_test_set[example_id]

    difference_array = np. subtract(true_values, predictions)
    squared_array = np. square(difference_array)
    mse = squared_array. mean()
    print('MSE first example: ', mse)

    linewidth = 1
    plt.plot(range(len(predictions)), predictions, color="red", label="predictions", linewidth=linewidth)
    plt.plot(range(len(true_values)), true_values, color="blue", label="true values", linewidth=linewidth)
    plt.legend()
    plt.savefig(os.path.join('data/test', 'singleTestSampleSetRezPerNodeID.png'))
    plt.clf()


predict()
