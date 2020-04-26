from feature_collection import *
from model_update import *

class LinearFeatureCollection(FeatureCollection):
    """
    Straight-up pass-through linear features
    plus a constant term
    """
    def __init__(self, reg_factor):
        self.dimension = 0
        self.reg_factor = reg_factor

    def set_dimension(self, dim):
        result = ModelUpdate(self.dimension + 1, dim + 1, self.reg_factor)
        self.dimension = dim
        return result

    def set_num_data_points(self, num):
        return None

    def get_num_features(self):
        return self.dimension + 1

    def get_features(self, v):
        return np.vstack(([1], v))
