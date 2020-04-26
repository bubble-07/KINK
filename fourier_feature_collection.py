import numpy as np
import math
import scipy as sp
from feature_collection import *

class FourierFeatureCollection(FeatureCollection):

    def __init__(self, reg_factor, feature_sampler, feature_num_determiner):
        #feature_sampler: function from num input dimensions, num_data_points to sampled angular velocity vecs
        #Feature determiner: function from num_dimensions, num_data points -> num randomized features

        self.num_features = 0

        #Collection of angular velocity vectors
        #which have been sampled
        #each row is a new angular velocity vector
        self.ws = np.array([[]])

        self.feature_sampler = feature_sampler
        self.feature_num_determiner = feature_num_determiner
        self.reg_factor = reg_factor

    def get_desired_num_features(self):
        return self.feature_num_determiner(self.num_dims, self.num_points)

    def get_num_features(self):
        return self.ws.shape[0]

    def get_features_dimension(self):
        return self.get_num_features() * 2

    def set_dimension(self, dim):
        self.num_dims = dim
        self.update_dimension()
        return self.create_more_features_if_needed()

    def update_dimension(self):
        #Truncates vectors in w or expands with zeros
        #as necessary to match the current number of dimensions
        n, d = self.ws.shape
        if (self.num_dims > d):
            new_ws = np.zeros((n, self.num_dims))
            new_ws[:n, :d] = self.ws
            self.ws = new_ws
        else if (self.num_dims < d):
            self.ws = self.ws[:n, :self.num_dims]

    def set_num_data_points(self, num):
        self.set_num_data_points = num
        return self.create_more_features_if_needed()

    def create_more_features_if_needed(self):
        desired_num_features = get_desired_num_features()
        actual_num_features = get_num_features()

        diff = desired_num_features - actual_num_features

        if (diff > 0):
            #If this happens, we need to add some features
            added_ws = np.zeros((diff, self.num_dims))
            for i in range(diff):
                new_w = self.feature_sampler(self.num_dims, self.set_num_data_points)
                added_ws[i] = new_w
            self.ws = np.vstack((self.ws, added_ws))
            #Great, now we just need to create the model update with the new info
            return ModelUpdate(actual_num_features, desired_num_features, self.reg_factor)

    def get_features(self, v):
        dotted = np.matmul(self.ws, v)
        cos_vec = np.cos(dotted)
        sin_vec = np.sin(dotted)

        result = np.zeros(self.get_features_dimension())
        result[::2] = cos_vec
        result[1::2] = sin_vec

        return result

