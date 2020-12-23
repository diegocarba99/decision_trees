import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from constants import *
from helper_functions import *
from node import *
from dataset import *
import numpy as np
import math
import os
from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter, DotExporter

"""
Class that holds a Decision Tree that can follow multiple algorithms (ID3, C4.5)
"""


class DecisionTree:
    """
	@attr   _train_set          dataset containing the training set
	@attr   _target             index of the feature that is the target
	@attr	_features           list containing the features from the dataset
	@attr	_features_values    list containing for each feature in the dataset,
	                            a list with it's possible values.
	@attr   _algorithm          indicates which algorithm the tree follows
	@attr   _root_node          root node of the decision tree
	@attr	_is_trained         indicates if the tree is trained or not
	@attr	_metrics            list containing the evaluation metrics
	@attr   _continious         list indicating if the features are continious
	"""

    __slots__ = ["_train_set", "_target", "_features", "_features_values",
                 "_algorithm", "_root_node", "_is_trained", "_metrics",
                 "_continious", "_conf_matrix"]

    def __init__(self, train_set, target, features, continious, values,
                 algorithm=0):
        """
		Creates and initializes a Decision Tree algorithm.

		@param	train_set	Training set which will be used to train the model.
		@param	target		Target value which the model has to predict.
		@param	features	Features lists.
		@param	algorithm	The algorithm the decision tree will follow.
		"""

        self._train_set = train_set
        self._target = target
        self._features = features
        self._features_values = values
        self._algorithm = algorithm
        self._root_node = Node("DECISION TREE", edge="DECISION TREE")
        self._is_trained = False
        self._metrics = None
        self._conf_matrix = [0, 0, 0, 0]
        self._continious = continious

    ################################################################################
    ############################## PUBLIC METHODS ##################################
    ################################################################################

    def train(self):
        """
        Trains the Decision Tree, depending on the selected algorithm.
        """
        # feature_set = set(self._features)
        feature_set = list(range(0, self._train_set.get_num_atributos()))
        feature_set.remove(self._target)
        self._root_node = self._tree_growing(self._train_set, feature_set,
                                             self._root_node)
        self._is_trained = True
        return self._root_node

    def predict(self, test_set):
        """
        Predicts the target values of the given test set following the trained
        tree

        @param	test_set	Test set data which has to be categorized
        @returns	Array of same size as test set's sample count with the
                    predicted target values
        """
        predicted_y = []
        missing_value_flag = True
        val = None

        for sample in test_set.get_data():
            node = self._root_node.children[0]

            while node.type != LEAF_NODE and missing_value_flag:
                branches = [n.edge for n in node.children]
                val = sample[node.feature]

                if val in branches:  # Known value, travel the tree
                    index = branches.index(val)

                else:  # Unknown value. select most common branch
                    types = [n.type for n in node.children]
                    if LEAF_NODE in types:
                        index = types.index(LEAF_NODE)
                    else:
                        missing_value_flag = False
                        if len(node.sample_count[0]) == 1:
                            val = node.sample_count[0][0]

                        elif node.sample_count[1][0] < node.sample_count[1][1]:
                            val = node.sample_count[0][1]

                        else:
                            val = node.sample_count[0][0]

                        # sample_count = [n.sample_count for n in node.children]
                        # index = sample_count.index(max(sample_count))

                node = node.children[index]

            if not missing_value_flag:
                missing_value_flag = True
            else:
                val = node.value

            if val == sample[-1]:
                predicted_y.append(True)

            else:
                predicted_y.append(False)

        return np.array(predicted_y)

    def compare_leave_one_out(self, test_y, predicted_y, confusion):
        """
        Compares the ground truth values of th training set to the predicted
        values for the target feature

        @param	test_y		Ground Truth data for the test set. numpy array
        @param	predicted_y	Predicted values for th target feature calculated
                            from the test set. numpy array
        """
        if test_y.shape != predicted_y.shape:
            error(f'predicted_y{predicted_y.shape} and test_y{test_y.shape} '
                  f'have not the same size and dimensions  ')
            exit()

        for ty, py in zip(test_y, predicted_y):
            if ty == 'won' and py:
                confusion[TP] += 1
            elif ty == 'won' and not py:
                confusion[FN] += 1
            elif ty == 'nowin' and py:
                confusion[FP] += 1
            elif ty == 'nowin' and not py:
                confusion[TN] += 1

        print(confusion)
        return confusion
        # return self._metrics, self._conf_matrix


        # tp = self._conf_matrix[TP]
        # fn = self._conf_matrix[FN]
        # fp = self._conf_matrix[FP]
        # tn = self._conf_matrix[TN]

        # print(self._conf_matrix)
        # self._metrics[ACCURACY] = (tp + tn) / (tp + tn + fp + fn)
        # self._metrics[PRECISION] = tp / (tp + fp)
        # self._metrics[RECALL] = tp / (tp + fn)
        # self._metrics[SPECIFICITY] = tn / (tn + fp)

    def calculate_metrics(self, confusion):

        tp = confusion[TP]
        fn = confusion[FN]
        fp = confusion[FP]
        tn = confusion[TN]
        self._metrics = [None] * 4

        self._metrics[ACCURACY] = (tp + tn) / (tp + tn + fp + fn)
        self._metrics[PRECISION] = tp / (tp + fp)
        self._metrics[RECALL] = tp / (tp + fn)
        self._metrics[SPECIFICITY] = tn / (tn + fp)

        return self._metrics


    def compare(self, test_y, predicted_y):
        """
        Compares the ground truth values of th training set to the predicted
        values for the target feature

        @param	test_y		Ground Truth data for the test set. numpy array
        @param	predicted_y	Predicted values for th target feature calculated
                            from the test set. numpy array
        """
        if test_y.shape != predicted_y.shape:
            error(f'predicted_y{predicted_y.shape} and test_y{test_y.shape} '
                  f'have not the same size and dimensions  ')
            exit()

        self._metrics = [None] * 4
        self._conf_matrix = [0, 0, 0, 0]

        for ty, py in zip(test_y, predicted_y):
            if ty == 'won' and py:
                self._conf_matrix[TP] += 1
            elif ty == 'won' and not py:
                self._conf_matrix[FN] += 1
            elif ty == 'nowin' and py:
                self._conf_matrix[FP] += 1
            elif ty == 'nowin' and not py:
                self._conf_matrix[TN] += 1

        tp = self._conf_matrix[TP]
        fn = self._conf_matrix[FN]
        fp = self._conf_matrix[FP]
        tn = self._conf_matrix[TN]

        self._metrics[ACCURACY] = (tp + tn) / (tp + tn + fp + fn)
        self._metrics[PRECISION] = tp / (tp + fp)
        self._metrics[RECALL] = tp / (tp + fn)
        self._metrics[SPECIFICITY] = tn / (tn + fp)

        return self._metrics, self._conf_matrix

    def print_confusion_matrix(self):
        print("--- CONFUSION MATRIX ------------------------------------------")
        print(f'TP:{self._conf_matrix[TP]} | FN:{self._conf_matrix[FN]}')
        print(f'FP:{self._conf_matrix[FP]} | TN:{self._conf_matrix[TN]}')

    def plot_confusion_matrix(self):
        grid = np.array(self._conf_matrix).reshape(2, 2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Reality")
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        im = ax.imshow(grid, interpolation='none', aspect='auto')

        labels = ["TP", "FN", "FP", "TN"]

        for (j, i), label in np.ndenumerate(grid):
            ax.text(i, j, f'{labels[i + j]}:{label}', ha='center', va='center')

        fig.colorbar(im)
        plt.plot()

    def print_metrics(self):
        print("--- METRICS ---------------------------------------------------")
        print(f'Accuracy: {self._metrics[ACCURACY]}')
        print(f'Precision: {self._metrics[PRECISION]}')
        print(f'Recall: {self._metrics[RECALL]}')
        print(f'Specificity: {self._metrics[SPECIFICITY]}')

    def print(self):
        """
        Print a representation of the tree in the standard output. The
        representation is done printing the nodes in a tabulated format.
        """
        for pre, fill, node in RenderTree(self._root_node.children[0]):
            # Feature node where tree splits. Print feature name
            if node.type == 0:
                print("%s(%s)-->[%s:%s]" % (pre, node.edge, node.feature, self._features[node.feature]))

            # Leaf node. Print the value of the target result
            elif node.type == 1:
                print("%s(%s)-->{%s}" % (pre, node.edge, node.value))

            # Feature's value node. Just a continuation for clearer view
            else:
                print("%sval:[%s]" % (pre, node.value))

    def to_picture(self):

        try:
            DotExporter(self._root_node.children[0]).to_picture(
                f'{TREE_IMAGE}.png')

        except Exception as e:
            error("Could not export the tree to a image. Please try installing "
                  "graphviz with the following command\n   sudo apt-get install "
                  "graphviz -y")
            return

    ################################################################################
    ############################## PRIVATE METHODS #################################
    ################################################################################

    def _same_target_samples(self, train_set):
        """
        Indicates if all the target features in a training set have the same
        value.

        @param  train_set   Training set to be analyzed
        @return True iff all the target values in training set are equal.
                False if not.
        """

        return True if len(np.unique(train_set.get_data()[-1])) == 1 else False

    def _stop_criterion(self, train_set, feature_set):
        """
        Indicates if the branch the tree is at that moment, has to stop growing
        and be tagged with a leaf.

        @param	train_set		Data from the training set
        @param	feature_set     Features that must be taken into consideration
        """

        if not len(feature_set):  # No features left
            return True

        if len(train_set.get_data()) == 0:  # If train set has samples
            return True

        if self._same_target_samples(train_set):  # All samples with same target
            return True

        return False

    def _split_criterion(self, train_set, feature_set, level):
        """
        Splits the tree depending on the dividing criteria, which is given by
        the algorithm.

        @param	train_set		Data from the training set
        @param	active_features	Features that must be taken into consideration
        @return	node which the tree will by divided at
        """
        split_feature = -1
        entropy_set = self._entropy_set(train_set)
        entropy_features = self._entropy_all_features(train_set, feature_set)
        gains = self._gain(entropy_set, entropy_features)

        if self._algorithm == ID3:
            split_feature = self._best_feature(gains, feature_set)

        else:  # Algorithm = C4.5
            splits = self._split_info_all_features(train_set, feature_set)
            gain_ratios = self._gain_ratio(gains, splits)
            split_feature = self._best_feature(gain_ratios, feature_set)

        unique, counts = np.unique(train_set.get_data()[:,-1], return_counts=True)

        return Node(f'F[{split_feature}]-L{level}', feature=split_feature,
                    feature_name=self._features[split_feature], type=0,
                    values=self._features_values[split_feature],
                    sample_count=[unique, counts])

    def _tree_growing(self, train_set, feature_set, raiz, level=0, val=None):
        """
        Recursive function that constructs a decision tree starting at a root
        node depending on the given training set and feature set.

        @param  train_set   training set from which the tree must be constructed
        @param  feature_set set containing the features that must be taken into
                            account
        @return the root node of the constructed decision tree
        """

        if self._stop_criterion(train_set, feature_set):  # Stop recursive calls
            # Tag the node as a leaf with the most common target value
            mct = self._most_common_target(train_set)
            Node(f'T[{mct}]-F[{raiz.feature}]-V[{val}]-L{level}', value=mct,
                 type=1, parent=raiz, edge=val)

        else:  # Keep iterating and constructing the Decision Tree
            # Create node where three splits according to the algorithm
            t = self._split_criterion(train_set, feature_set, level)
            t.edge = val
            t.parent = raiz  # Connect created node to the parent node
            feature_set.remove(t.feature)  # Remove splitting feature from set

            for v_i in t.values:  # For every possible value of the feature
                # Remove samples that don't contain v_i as value for the feature
                trimmed_train_set = self._trim_dataset(train_set, feature_set,
                                                       t.feature, v_i)

                if len(trimmed_train_set.get_data()) > 0:
                    # Recursive call. Keep calculating the tree
                    self._tree_growing(trimmed_train_set, feature_set, t,
                                       level=level + 1, val=v_i)

        return raiz

    def _most_common_target(self, train_set):
        """
        Given a dataset, gets the value of the target feature that appears most
        taking into account all the samples of the dataset

        @param  train_set   dataset where the target values will be analized
        @return value of the target feature that appears most in the dataset
        """

        unique, counts = np.unique(train_set.get_data()[:, -1],
                                   return_counts=True)
        return unique[np.argmax(counts)]

    def _entropy_set(self, train_set):
        """
        Calculates the entropy of the given training set
        """

        entropy = 0
        s_count = len(train_set.get_data())  # |S|

        unique, counts = np.unique(train_set.get_data()[-1], return_counts=True)

        for target_count in counts:

            if target_count > 0:
                p_x = target_count / s_count  # p(x)
                entropy += p_x * math.log(p_x, 2)  # p(x)*log_2(p(x))

        return -entropy

    def _entropy_set_nparray(self, train_set):
        """
        Calculates the entropy of the given training set, where the set is not
        an object of the Database class but a numpy array just containing the
        data of a train_set
        """

        entropy = 0
        s_count = len(train_set)  # |S|

        unique, counts = np.unique(train_set[-1], return_counts=True)

        for target_count in counts:

            if target_count > 0:
                p_x = target_count / s_count  # p(x)
                entropy += p_x * math.log(p_x, 2)  # p(x)*log_2(p(x))

        return -entropy

    def _entropy_feature(self, train_set, f, feature_set):
        """
       	Calculates the entropy of a given feature compared to the whole training
       	set
        """

        entropy = 0
        s_count = len(train_set.get_data())  # |S|

        for v_i in self._features_values[f]:
            set_v_i = self._trim_dataset(train_set, feature_set, f, v_i)  # S_v
            sv_count = len(set_v_i.get_data())  # |S_v|
            if sv_count == 0:  # log_2(0) cannot be done
                continue
            entropy += (sv_count / s_count) * self._entropy_set(set_v_i)

        return entropy

    def _split_info_all_features(self, train_set, feature_set):
        split_info = []

        for f in feature_set:
            # if self._continious[f]:
            #    pass
            # else:
            split_info.append(
                self._split_info_feature(train_set, feature_set, f))

        return split_info

    def _split_info_feature(self, train_set, feature_set, f):
        split = 0
        s_count = len(train_set.get_data())  # |S|

        for v_i in self._features_values[f]:
            set_v_i = self._trim_dataset(train_set, feature_set, f, v_i)
            sv_count = len(set_v_i.get_data())  # |S_v|

            if sv_count == 0:  # log_2(0) cannot be done
                continue

            split += (sv_count / s_count) * math.log(sv_count / s_count, 2)

        return -split

    def _gain_ratio(self, gains, splits):
        gain_ratios = []

        for i in range(len(gains)):

            if splits[i] != 0:
                gain_ratios.append(gains[i] / splits[i])

            else:
                gain_ratios.append(0)

        return gain_ratios

    def _entropy_all_features(self, train_set, feature_set):
        """
        Calculates the entropy of all the features compared to the whole
        training set
        """

        entropies = []

        for f in feature_set:
            # if self._continious[f]:
            #    pass
            # else:
            entropies.append(self._entropy_feature(train_set, f, feature_set))

        return entropies

    def _gain(self, entropy_set, entropy_features):
        """
        Returns a list with all the gains of each active features for the whole
	    train set
        """

        gains = []
        for elem in entropy_features:
            gains.append(entropy_set - elem)
        return gains

    def _best_feature(self, info, feature_set):
        """
        Given a list with the gains for all the features, returnes the feature
        that maximizes the gain

        @param  gains       list with the gains for all the features
        @param  feature_set set with all the active features
        @return the feature that has the biggest gain
        """
        return feature_set[info.index(max(info))]

    def _trim_dataset(self, dataset, features, index, val):
        """
        Trims a given dataset to only contain samples that have a certain value
        for a certain feature in the set

        @param  dataset dataset to be trimmed
        @param  index   index of the feature that must have a certain value
        @param  val     value of the feature to be trimming by
        @return same horizontal dimension trimmed input dataset
        """
        arr = np.where(dataset.get_data()[:, index] == val, True, False)
        i = 0
        rows_to_delete = []
        for elem in arr:
            if not elem:
                rows_to_delete.append(i)
                i += 1

        # debug2(f'trimming {len(rows_to_delete)} samples')

        trimmed_data = np.delete(dataset.get_data(), rows_to_delete, 0)
        debug2(f'lenght of trimmed dataset is 0') if len(
            trimmed_data) == 0 else None
        return Dataset(trimmed_data, features)

    ############################################################################
    ############################ GETTERS AND SETTERS ###########################
    ############################################################################

    def get_target(self):
        return self._target

    def get_accuracy(self):
        """
        Get the accuracy from the comparison between the prediction and the
	    ground truth
        """

        if not self._is_trained:
            print("ERROR: Fit the classifier before obtaining metrics")

        elif self._metrics is None:
            print("ERROR: Compare some predictions in order to get the metrics")

        else:
            return self._metrics[ACCURACY]

    def get_precision(self):
        """
        Get the precision from the comparison between the prediction and the
	    ground truth
        """

        if not self._is_trained:
            error("Fit the classifier before obtaining metrics")

        elif self._metrics is None:
            error("Compare some predictions in order to get the metrics")

        else:
            return self._metrics[PRECISION]

    def get_recall(self):
        """
        Get the recall from the comparison between the prediction and the ground
        truth
        """

        if not self._is_trained:
            error("Fit the classifier before obtaining metrics")

        elif self._metrics is None:
            error("Compare some predictions in order to get the metrics")

        else:
            return self._metrics[RECALL]

    def get_specificity(self):
        """
        Get the sensitivity from the comparison between the prediction and the
	    ground truth
        """
        if not self._is_trained:
            print("ERROR: Fit the classifier before obtaining metrics")

        elif self._metrics is None:
            print("ERROR: Compare some predictions in order to get the metrics")

        else:
            return self._metrics[SPECIFICITY]

    def get_all_metrics(self):
        """
        Get all the metrics from the comparison between the prediction and the
	    ground truth
        """

        if not self._is_trained:
            print("ERROR: Fit the classifier before obtaining metrics")

        elif self._metrics is None:
            print("ERROR: Compare some predictions in order to get the metrics")

        else:
            return self._metrics

    def set_metrics(self, metrics):
        self._metrics = metrics

    def set_confusion(self, confusion):
        self._conf_matrix = confusion

    def get_tree(self):
        """
        @return trained decision tree
        """
        return self._root_node

    def set_algorithm(self, algorithm):
        """
        Sets the algorithm the tree will follow. Possible values:
            0 - ID3
            1 - C4.5
            2 - Random Forest

        @param	algorithm	Index of the algorithm the tree has to follow
        """

        self._algorithm = algorithm
