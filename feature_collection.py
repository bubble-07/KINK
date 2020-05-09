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

    def __init__(self, reg_factor):
        self.reg_factor = reg_factor
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

    def blank_mean(self, out_dims, update=False):
        return np.zeros((out_dims, self.get_features_dimension()))

    def blank_precision(self, other_feature_collection, out_dims, update=False):
        t = out_dims
        s = self.get_features_dimension()
        s_other = other_feature_collection.get_features_dimension()

        if (update or other_feature_collection != self):
            result = np.zeros((t, s, t, s_other))
        else:
            #By default, just yield kroneckered diagonals weighted appropriately
            result = self.blank_precision_diagonal(s, t)
        return result

    def blank_precision_diagonal(self, s, t):
        diagonal_mat = np.kron(np.eye(t), np.eye(s))
        result = np.reshape(diagonal_mat, (t, s, t, s))
        result *= self.reg_factor * self.get_features_dimension()
        return result

    def adjust_mean_output_dims(self, prior_mean, new_output_dims, update=False):
        """
        Given the t_0 x s mean sub-matrix for a model on this feature
        collection and a new number of output
        dims [updates t], returns an updated mean sub-matrix of size t_1 x s
        """
        #Default: just pad the mean with zeroes
        t, s = prior_mean.shape
        result = prior_mean.copy()
        result.resize((new_output_dims, s))
        return result
    
    def adjust_precision_output_dims(self, other_feature_collection, prior_precision, new_output_dims, update=False):

        """
        Given the (t_0 x s) x (t_0 x s_other) precision sub-tensor for a model and a new
        number of output dims [updates t], returns an updated precision
        tensor of shape (t_1 x s) x (t_1 x s_other)
        """
        result = self.blank_precision(other_feature_collection, new_output_dims, update)
        utils.copy_from_into_4D(prior_precision, result)
        return result
