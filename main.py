#!/usr/bin/python

import matplotlib.pyplot as plt
from dataset import *
from constants import *
from decisiontree import *
from reader import *
from helper_functions import *
from anytree.exporter import DotExporter
import numpy as np

if __name__ == "__main__":

    info("Decision Tree Classifier:")
    debug("Creating reader class with route to the data folder")
    reader = Reader(DATA_FOLDER_DIR)
    done()

    debug("Reading data from  file")
    raw_dataset = reader.read_data(DATA_FILENAME)
    done()
    info(f'Dataset data: shape: {raw_dataset.shape}\tsize: {raw_dataset.size}')

    debug("Reading attributes description from file")
    raw_attributes = reader.read_attributes(NAME_FILENAME)
    done()
    info(f'Dataset features: shape: {raw_attributes.shape}\tsize: {raw_attributes.size}')

    debug("Creating main dataset object wth raw data")
    dataset = Dataset(raw_dataset, raw_attributes)
    done()
    info(dataset.to_string())

    tree = None

    if SELECTED_SPLIT == HOLDOUT:
        info("Proceeding with holdout algorithm")

        debug("Creating train-dev-test dataset divisions")
        datasets = dataset.create_sets(HOLDOUT)
        done()

        # Calculate the target features index, and get the features descriptions
        target_index = datasets[TRAIN].get_num_atributos() - 1
        features = datasets[TRAIN].get_atributos()
        cont = datasets[TRAIN].get_continuos()
        values = datasets[TRAIN].get_valores_atributos()

        debug("Creating Decision tree object with the training data")
        tree = DecisionTree(datasets[TRAIN], target_index, features, cont,
                            values, SELECTED_ALGO)
        done()

        debug("Training decision tree")
        tree.train()
        done()

        debug("Predicting target values from test set")
        predicted_y = tree.predict(datasets[TEST])
        done()

        debug("Comparing predicted results to ground truth")
        tree.compare(datasets[TEST].get_data()[:,-1], predicted_y)
        done()

        # Print performance metrics
        tree.print_metrics()
        tree.plot_confusion_matrix()
        tree.print_confusion_matrix()
        # Print the decision tree
        tree.print()
        tree.to_picture()

    elif SELECTED_SPLIT == CV:
        info("Proceeding with cross-validation algorithm")

        debug("Creating train-dev-test dataset divisions")
        datasets = dataset.create_sets(CV, k=K)
        features = dataset.get_atributos()
        done()

        train = None
        test = None
        acum_metrics = [0, 0, 0, 0]
        acum_conf = [0, 0, 0, 0]


        for k in range(0, K):

            k_train = Dataset(datasets[k][0], features)
            k_test = Dataset(datasets[k][1], features)

            # Calculate the target features index, and get the features descriptions
            target_index = k_train.get_num_atributos() - 1
            features = k_train.get_atributos()
            cont = k_train.get_continuos()
            values = k_train.get_valores_atributos()

            debug(f'{K}-fold with K={k} - Creating Decision tree object with the training data')
            tree = DecisionTree(k_train, target_index, features, cont,
                                values, SELECTED_ALGO)
            done()

            debug(f'{K}-fold with K={k} - Training decision tree')
            tree.train()
            done()


            debug(f'{K}-fold with K={k} - Predicting target values from test set')
            predicted_y = tree.predict(k_test)
            done()

            debug(f'{K}-fold with K={k} - Comparing predicted results to ground truth')
            tree_metrics, tree_confusion = tree.compare(k_test.get_data()[:, -1], predicted_y)
            done()

            for i in range(0, 4):
                acum_metrics[i] += tree_metrics[i]
                acum_conf[i] += tree_confusion[i]

        info(f'Cross-validation with {K}-fold metrics\' arithmetic mean results')
        tree.set_metrics([e/K for e in acum_metrics])
        tree.set_confusion([e/K for e in acum_conf])
        # Print performance metrics
        tree.print_metrics()
        tree.plot_confusion_matrix()
        tree.print_confusion_matrix()
        # Print the decision tree
        #tree.print()
        #tree.to_picture()

    elif SELECTED_SPLIT == LEAVE_ONE_OUT:
        info("Proceeding with leave-one-out algorithm")

        debug("Creating train-dev-test dataset divisions")
        datasets = dataset.create_sets(LEAVE_ONE_OUT)
        features = dataset.get_atributos()
        cont = dataset.get_continuos()
        done()
        acum_metrics = [0, 0, 0, 0]
        acum_conf = [0, 0, 0, 0]
        n = len(datasets[0].get_data())
        confusion = [0, 0, 0, 0]


        for k in range(n):
            split = np.vsplit(datasets[0].get_data(), np.array([k, k+1]))
            t_slice = np.vstack((split[0], split[2]))
            k_train = Dataset(t_slice, features, cont=cont)
            k_test = Dataset(split[1], features, cont=cont)

            # Calculate the target features index, and get the features descriptions
            target_index = k_train.get_num_atributos() - 1
            features = k_train.get_atributos()
            cont = k_train.get_continuos()
            values = k_train.get_valores_atributos()

            debug(f'Leave-one-out step {k} - Creating Decision tree object with the training data')
            tree = DecisionTree(k_train, target_index, features, cont,
                                values, SELECTED_ALGO)
            done()

            debug(f'Leave-one-out step {k} - Training decision tree')
            tree.train()
            done()

            debug(f'Leave-one-out step {k} - Predicting target values from test set')
            predicted_y = tree.predict(k_test)
            done()

            debug(f'Leave-one-out step {k} - Comparing predicted results to ground truth')
            confusion = tree.compare_leave_one_out(k_test.get_data()[:, -1], predicted_y, confusion)
            done()


        info(f'Leave-one-out step {k} - metrics\' arithmetic mean results')
        tree.calculate_metrics(confusion)
        # Print performance metrics
        tree.print_metrics()
        tree.plot_confusion_matrix()
        tree.print_confusion_matrix()
        # Print the decision tree
        tree.print()
        tree.to_picture()
    else:
        error("Selected algorithm not recognised. Review 'constants.py' file")

    # # Print performance metrics
    # tree.print_metrics()
    # tree.plot_confusion_matrix()
    # # Print the decision tree
    # tree.print()
    # tree.to_picture()
