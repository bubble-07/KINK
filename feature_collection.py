import numpy as np
import math
import scipy as sp
import scipy.spatial.distance

class FeatureCollection(object):
    """
    Manages a collection of features
    for some RKHS, with the ability
    to dynamically modify the dimension
    of the base space and the number of samples
    taken in the base space
    """

    def __init__(self):
        pass

    def set_dimension(self, dim):
        """
        Sets the dimension of the base space.
        Possibly returns a ModelUpdate
        """
        pass

    def set_num_data_points(self, num):
        """
        Sets the number of data points in the base space
        Possibly returns a ModelUpdate
        """
        pass

   def get_features_dimension(self):
       """
       Gets the dimension of the feature space
       May be different from the number of features
       """
       return self.get_num_features()

   def get_num_features(self):
        """
        Gets the current number of features in the feature
        vectors returned by this FeatureCollection
        """
        pass
    
   def get_features(self, v):
        """
        Converts a vector in the base space to a vector in
        the RKHS feature space
        """
        pass

