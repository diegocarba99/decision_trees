import numpy as np
import pandas as pd

'''
Class that handles the reading operations to retrieve the necessary information 
from the database files.
'''


class Reader(object):
    """
    @attr   _folder     folder where all the files related to the database are
                        located
    """
    __slots__ = ["_folder"]

    def __init__(self, folder):
        """
        Initializes a dataset reader from a folder source. The file should be
        located in this folder to be found.

        @param folder   folder path where all the files are located
        """
        self._folder = folder

    def read_data(self, filename):
        """
        Read the file where the data for the model is. Stores all the data from the
        file in a numpy matrix. Rows in the matrix represent each sample from the
        database and each column represents an attribute. Last column represents the
        target feature.
        """
        file = pd.read_csv(self._folder + filename)
        return np.array([data for data in file.values])

    def read_attributes(self, filename):
        """
        Read the file where the attributes names are. Stores all the attribute names
        in a list. The names are stores in a file where each line corresponds to a
        new attribute name.

        @param  filename    name of the file where attribute names are stored
        @return list containing all the attribute names from 'filename'
        """
        return np.array([line[:-1] for line in open(self._folder + filename)])
