import numpy as np
import random
from constants import *
from helper_functions import *

"""
Class that encapsulates a sample database wih all it's attributes. The 
information is stored in a matrix, and relevant information about the data is 
stored for it's future use
"""


class Dataset:
    """
    @attr   _data               matrix containing the data
    @attr	_x					matrix data without the target feature
    @attr	_y					column matrix containing the target feature
    @attr   _num_muestras	  	sample count of the dataset
    @attr   _num_atributos	 	feature coutn of the dataset
    @attr   _atributos		    list containing a description of the features
    @attr   _valores_atributos  list containing for each feature in the dataset,
                                a list with it's possible values.
    @attr   _continuo		    list indicating if the features are continious
    """
    __slots__ = ["_data", "_num_muestras", "_num_atributos", "_atributos",
                 "_valores_atributos", "_continuo", "_x", "_y"]

    """
    Initializes a new data set with the data given, setting the number of items 
    and features manually, or either being automatic (then len of data are rows 
    and len of data[0] will be cols). If continous values exist in the dataset,
    a list indicating which features are continous must be provided through the
    'continuo' option.

    @param  data		the data of the datset
    @param  cont    	lista booleana indicando que atributos son continuos
    """

    def __init__(self, data, atributos, cont=None):
        self._data = data
        #self._x = self._data[:-1]
        #self._y = self._data[-1]
        self._atributos = atributos
        self._continuo = cont
        self._num_muestras = len(data)
        self._num_atributos = len(data[0]) if len(data) else 0
        self._filter_missing_values()
        self._calcular_valores_atributos()
        if cont is None:
            self._continuo = np.zeros(self._num_atributos, dtype=bool)

    def create_sets(self, split_method, percent=0.8, k=5):
        """
        Given a dataset, constructs the training, test and if needed validation
        datasets to train a model. The way these datasets are created deppends
        on the split_method, which indicates which of the following methods
        follow:
            - Holdout
            - Cross validation
            - Leave one out
            - Bootstrap
        """

        data = self._data
        np.random.seed(SEED) if SHUFFLE else None
        np.random.shuffle(data) if SHUFFLE else None
        datasets = []

        if split_method == HOLDOUT:
            assert percent > 0.0, error(
                "Holdout percent should be greater than "
                "0%")

            split_number = int(self._num_muestras * percent)
            datasets.append(Dataset(data[:split_number], self._atributos,
                                    cont=self._continuo))  # train set
            datasets.append(Dataset(data[split_number:], self._atributos,
                                    cont=self._continuo))  # test set

        elif split_method == CV:
            n = self._num_muestras // k
            for i in range(0, k):
                start = i * n
                end = self._num_muestras if i == k - 1 else (i + 1) * n
                split = np.vsplit(data, np.array([start, end]))
                t_slice = np.vstack((split[0], split[2]))
                train = Dataset(t_slice, self._atributos, cont=self._continuo)
                test = Dataset(split[1], self._atributos, cont=self._continuo)

                datasets.append([np.vstack((split[0], split[1])), split[1]])

        else:  # LEAVE-ONE-OUT
            datasets.append(Dataset(data, self._atributos, cont=self._continuo))

        return datasets

    def _calcular_valores_atributos(self):
        """
        Calculates all the possible values for every feature and stores it in a
        list

        @return list indicating all the possible values for all the features
        """
        self._valores_atributos = [
            sorted(list(set([muestra[atributo] for muestra in self._data]))) for
            atributo in range(self._num_atributos)]


    def _filter_missing_values(self):
        d = self._data
        for i in range(self._num_atributos):
            if '?' in d[:, i]:
                unique, counts = np.unique(d[:, -1], return_counts=True)
                mcv = unique[np.argmax(counts)]
                d[:, i] = np.where(d[:, i] == '?', mcv, d[:, i])
        self._data = d



    def get_data(self):
        """
        Get the whole dataframes data
        """
        return self._data

    def get_atributos(self):
        """
        Get the atributes of the dataset
        """
        return self._atributos

    def get_num_muestras(self):
        """
        Get the number of samples in the dataframe
        """
        return self._num_muestras

    def get_num_atributos(self):
        """
        Get the number of features in the dataframe
        """
        return self._num_atributos

    def get_continuos(self):
        """
        Get the list that indicates if the features are continious or not
        """
        return self._continuo

    def get_continuos_feature(self, feature):
        """
        Indicates if a feature of the dataset is continious or not
        """
        return self._continuo[feature]

    def get_valores_atributos(self):
        """
        Get all the possible values for all the feature in the dataset
        """
        return self._valores_atributos

    def get_valores_atributo(self, atributo_index):
        """
        Get all the possible values for a certain feature
        """
        if atributo_index >= 0 and atributo_index < self._num_atributos:
            return self._valores_atributos[atributo_index]
        else:
            return []

    def get_x(self):
        """
        Returns the whole dataset without the target values column

        @return whole dataset without target values column as a pandas dataframe
        """
        return self._x

    def get_y(self):
        """
        Returns the target values column

        @return target values column as a pandas dataframe
        """
        return self._y

    def to_string(self):
        """
        Represents the database as a string

        @return string containing the textual representation of the database
        """

        txt = "Clase: database.py\n"
        txt += "  [X] Muestras:	 %d muestras\n" % (self._num_muestras)

        txt += "  [X] Atributos: %d atributos, %d continuos" % (
            self._num_atributos, np.count_nonzero(self._continuo == True))
        # txt += "  [X] Posibles valores para los atributos\n"
        # for i in range(self._num_atributos):
        #     txt += "     -  %s: %s\n" % (
        #         self._atributos[i], self._valores_atributos[i])
        # txt += "\n"
        return txt
