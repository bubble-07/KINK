import numpy as np
import math
from model import ModelSpace
import fourier_feature_collection as fourier
import linear_feature_collection as linear
import quadratic_feature_collection as quadratic

#Class for "the model", a particular instance of a model space
FOURIER_REG = 1.0
LINEAR_REG = 1.0
QUADRATIC_REG = 1.0

FREQ_SCALING = 10.0
QUAD_SCALING = 5.0
FOURIER_SCALING = 1.0

def get_num_fourier_features(d, n):
    return int(math.ceil(math.sqrt(n) * FOURIER_SCALING))

def fourier_feature_sampler(d, n):
    return np.random.standard_cauchy(d) * FREQ_SCALING

def get_num_quadratic_features(d, n):
    return d * QUAD_SCALING

linear_collection = linear.LinearFeatureCollection(LINEAR_REG)
quadratic_collection = quadratic.QuadraticFeatureCollection(QUADRATIC_REG, get_num_quadratic_features)
fourier_collection = fourier.FourierFeatureCollection(FOURIER_REG, fourier_feature_sampler, get_num_fourier_features)

linear_collection.set_dimension(1)
linear_collection.set_num_data_points(1)

quadratic_collection.set_dimension(1)
quadratic_collection.set_num_data_points(1)

fourier_collection.set_dimension(1)
fourier_collection.set_num_data_points(1)

collections = [linear_collection, quadratic_collection, fourier_collection]
#collections = [linear_collection, fourier_collection]
#collections = [linear_collection]
#collections = [quadratic_collection]
#collections = [fourier_collection]

class TheModelSpace(ModelSpace):
    def __init__(self):
        ModelSpace.__init__(self, collections)


