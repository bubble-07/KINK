from feature_collection import *
from model_update import *

class LinearFeatureCollection(FeatureCollection):
    """
    Straight-up pass-through linear features
    plus a constant term
    """
    def __init__(self, reg_factor):
        FeatureCollection.__init__(self, reg_factor)
        self.dimension = 0

    def set_dimension(self, dim):
        if (self.dimension == dim):
            return None
        result = ModelUpdate(self, self.dimension + 1, dim + 1, self.reg_factor)
        self.dimension = dim
        return result

    def set_num_data_points(self, num):
        return None

    def get_num_features(self):
        return self.dimension + 1

    def get_features(self, v):
        return np.concatenate(([1], v))
