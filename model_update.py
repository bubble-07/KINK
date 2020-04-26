import numpy as np

class ModelUpdate(object):
    def __init__(self, prev_num_features, current_num_features, reg_factor):
        self.prev_num_features = prev_num_features
        self.current_num_features = current_num_features
        self.reg_factor = reg_factor

    def get_prev_num_features(self):
        return self.prev_num_features

    def get_current_num_features(self):
        return self.current_num_features

    def update_mean(self, model, prev_mean):
        """
        Given a model with the full (outputs) t x s (inputs) model mean matrix,
        and the t x f_0 submatrix of the previous mean for these features
        [where f_0 is the previous number of features]
        yields a t x f_1 matrix block containing the updated slice of the
        model mean matrix, where f_1 is the new number of features
        """
        #Default implementation: Just pad with zeroes

        t, f_zero = prev_mean.shape
        assert get_prev_num_features() == f_zero
        return np.zeros((t, get_current_num_features()))

    def update_precision(self, model, prev_precision):
        """
        Given a model with the full (t x s) x (t x s) model precision tensor,
        and the (t x f_0) x (t x f_0) tensor slice containing the previous
        precision tensor for these features,
        yields a (t x f_1) x (t x f_1) tensor slice containing the updated
        model precision within this feature set, and another slice of size
        (t x f_1) x (t x r) where r = s - f_1 with the tensor slice containing
        any interaction terms in the precision between these features and
        the rest of the features in the model
        """

        #Default implementation: No interaction terms, just the diagonal
        t, f_zero, _, _ = prev_precision.shape
        f_one = get_current_num_features()
        _, s, _, _ = model.get_precision_tensor().shape
        r = s - f_one

        diagonal_mat = np.kron(np.eye(t), np.eye(f_one))
        diagonal = np.reshape(diagonal_mat, (t, f_one, t, f_one))

        diagonal = diagonal * self.reg_factor

        interaction = np.zeros((t, f_one, t, r))

        return (diagonal, interaction)



        


