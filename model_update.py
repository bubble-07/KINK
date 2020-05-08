import numpy as np
import utils

class ModelUpdate(object):
    def __init__(self, originating_feature_collection, prev_num_features, current_num_features, reg_factor):
        self.originating_feature_collection = originating_feature_collection
        self.prev_num_features = prev_num_features
        self.current_num_features = current_num_features
        self.reg_factor = reg_factor

    def get_prev_num_features(self):
        return self.prev_num_features

    def get_current_num_features(self):
        return self.current_num_features

    def update_mean(self, prev_mean, update=False):
        """
        Given a model with the full (outputs) t x s (inputs) model mean matrix,
        and the t x f_0 submatrix of the previous mean for these features
        [where f_0 is the previous number of features]
        yields a t x f_1 matrix block containing the updated slice of the
        model mean matrix, where f_1 is the new number of features
        """
        #Default implementation: Just pad with zeroes

        t, f_zero = prev_mean.shape
        assert self.get_prev_num_features() == f_zero
        return np.zeros((t, self.get_current_num_features()))

    def update_precision(self, other_feature_collection, prev_precision, update=False):
        """
        Given a model with the full (t x s) x (t x s) model precision tensor,
        and the (t x s0_init) x (t x s1) slice of the model precision tensor
        corresponding to the precision between the originating feature collection
        and other_feature_collection (with s0_init being the previous dimension of the feature
        space on the originating feature collection, and s1 being the dimension
        of the feature space on other_feature_collection),
        computes an updated (t x s0_final) x (t x s1) precision tensor.

        In the special case where other_feature_collection is the same as the originating
        feature collection, the result will instead be (t x s0_final) x (t x s0_final)
        """
        t, s_zero_init, _, s_one = prev_precision.shape
        s_zero_final = self.current_num_features

        #If this is just a bayesian update tensor we're adjusting, always fall through
        #to just zero-padding
        if (update == True):
            result = prev_precision.copy()
            result.resize((t, s_zero_final, t, s_one))
            return result

        #Otherwise, must be model parameters we're updating, so we need to pass along
        #info about the prior we're using

        if (other_feature_collection == self.originating_feature_collection):
            #This is the diagonal part of the whole shebang
            diagonal_mat = np.kron(np.eye(t), np.eye(s_zero_final))
            diagonal = np.reshape(diagonal_mat, (t, s_zero_final, t, s_zero_final))
            result = diagonal * self.reg_factor
            utils.copy_from_into_4D(prev_precision, result)
            return result
        else: 
            #otherwise, must be an interaction term. By default, just zero-pad
            result = prev_precision.copy()
            result.resize((t, s_zero_final, t, s_one))
            return result
 
