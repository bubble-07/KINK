import numpy as np
from feature_collection import *
from model_update import *
import numpy.fft as fft

class QuadraticFeatureCollection(FeatureCollection):
    """
    Quadratic features [homogenous quadratic polynomial kernel]
    using randomized tensor decomposition
    """
    def __init__(self, reg_factor, feature_num_determiner):
        """
        Inputs: regularization factor and a function which takes
        num_dimensions, num_data points -> num randomized features
        """
        self.reg_factor = reg_factor
        self.num_dims = 0
        self.num_points = 0
        self.feature_num_determiner = feature_num_determiner

        self.sketch_one = CountSketch()
        self.sketch_two = CountSketch()

    def get_desired_num_features(self):
        return self.feature_num_determiner(self.num_dims, self.num_points)

    def get_actual_num_features(self):
        return self.sketch_one.get_out_dims()

    def set_dimension(self, dim):
        self.num_dims = dim
        return self.issue_update_if_needed()

    def set_num_data_points(self, num):
        self.set_num_data_points = num
        return self.issue_update_if_needed()


    def issue_update_if_needed(self):
        desired_num_features = get_desired_num_features()
        actual_num_features = get_actual_num_features()
        if (desired_num_features > actual_num_features):
            #If this happens, we need to double the number of features
            self.sketch_one.double_out_dims()
            self.sketch_two.double_out_dims()
            new_num_features = get_actual_num_features()
            return QuadraticModelUpdate(actual_num_features, new_num_featuresj)

    def get_features(self, v):
        first_sketch = self.sketch_one.sketch(v)   
        second_sketch = self.sketch_two.sketch(v)
        #FFT polynomial multiplication
        first_fft = fft.fft(first_sketch)
        second_fft = fft.fft(second_sketch)
        multiplied_fft = first_fft * second_fft
        result = fft.ifft(multiplied_fft)
        return result

class CountSketch(object):
    def __init__(self):
        self.in_dims = 0
        self.out_dims = 1

        #An array with the size of the number
        #of input features containing indices
        #to corresponding output feature accumulation positions
        self.indices = np.array([])

        #An array with the size of the number of
        #input features containing signs (-1/+1)
        #to corresponding ouptut feature accumulation positions
        self.signs = np.array([])

        self.rng = np.random.default_rng()

    def sketch(self, v):
        signed_v = v * self.signs
        result = np.zeros(self.out_dims)
        np.add.at(result, self.indices, signed_v)
        return result

    def set_in_dims(self, in_dims):
        if (self.in_dims > in_dims):
            #Case: shrinking dimensions
            self.indices = self.indices[:in_dims]
            self.signs = self.signs[:in_dims]
            self.in_dims = in_dims
        else if (in_dims > self.in_dims):
            #Case: adding dimensions
            diff = in_dims - self.in_dims
            self.in_dims = in_dims

            extra_indices = self.rng.integers(self.out_dims, size=diff)
            extra_signs = self.rng.choice([-1, 1], size=diff)

            self.indices = np.concatenate((self.indices, extra_indices))
            self.signs = np.concatenate((self.signs, extra_signs))

    def double_out_dims(self):
        offset = self.out_dims
        self.out_dims *= 2
        #Randomly decide whether/not to add the offset
        #to each input coordinate
        offsets = self.rng.integers(2, size=self.in_dims) * offset
        self.indices += offsets

    def get_out_dims(self):
        return self.out_dims

    def get_in_dims(self):
        return self.in_dims

class QuadraticModelUpdate(ModelUpdate):
    def __init__(self, prev_num_features, current_num_features):
        super(self, prev_num_features, current_num_features, 0.0)

    def update_mean(self, model, prev_mean):
        #The mean was previously t x s, but we need to make it t x 2s
        #by tiling
        return np.hstack((prev_mean, prev_mean))

    def update_precision(self, model, prev_precision):
        #Don't care about interaction terms for now
        _, interaction_slice = super.update_precision(model, prev_precision)
        #The previous precision matrix comes in as a (t x s) x (t x s)
        #tensor -- we need to expand this to a (t x 2s) x (t x 2s) tensor
        #by putting stuff on the diagonal

        two_precision = prev_precision * 2

        zeroes = np.zeros_like(prev_precision)

        part_one = np.concatenate((two_precision, zeroes), axis=3)
        part_two = np.concatenate((zeroes, two_precision), axis=3)

        result = np.concatenate((part_one, part_two), axis=1)

        return result, interaction_slice
