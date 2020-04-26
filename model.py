import numpy as np

class Model(object):

    def __init__(self, feature_collections):
        self.feature_collections = feature_collections
        self.num_in_dims = 0
        self.num_out_dims = 0
        self.num_data_points = 0

    def set_num_out_dims(self, num_out_dims):
        diff = num_out_dims - self.num_out_dims
        if (num_out_dims > self.num_out_dims):
            _, s = self.mean_matrix.shape
            mean_matrix = self.mean_matrix
            precision_tensor = self.precision_tensor

            #Update the mean matrix
            self.mean_matrix = np.zeros((num_out_dims, s))
            self.mean_matrix[:self.num_out_dims, :] = mean_matrix

            #Update the precision tensor
            self.precision_tensor = np.zeros((num_out_dims, s, num_out_dims, s))
            self.precision_tensor[:self.num_out_dims, :, :self.num_out_dims, :] = precision_tensor
        
            #Update the number of output dims
            self.num_out_dims = num_out_dims

    def get_precision_tensor(self):
        """
        Gets the current value of the (t x s) x (t x s) precision tensor
        """
        return self.precision_tensor

    def get_mean_matrix(self):
        """
        Gets the current value of the t x s mean matrix
        """
        return self.mean_matrix

    def get_feature_end_index(self, feature_collection_index, feature_index):
        feature_collection = self.feature_collections[feature_collection_index]
        num_features = feature_collection.get_features_dimension()
        feature_end_index = feature_index + num_features
        return feature_end_index

    def get_collection_mean(self, feature_collection_index, feature_index):
        """
        Gets the mean sub-matrix for the given feature,
        given the index of the feature collection in the list
        of feature collections and the absolute index of
        the start of the features for the collection
        """
        feature_end_index = get_feature_end_index(feature_collection_index, feature_index)
        result = self.mean_matrix[:, feature_index:feature_end_index]
        return result

    def get_collection_precision(self, feature_collection_index, feature_index):
        feature_end_index = get_feature_end_index(feature_collection_index, feature_index)
        result = self.precision_tensor[:, feature_index:feature_end_index, :, feature_index:feature_end_index]
        return result

    def apply_model_update(self, update, feature_collection_index, feature_index):
        feature_collection_mean = get_collection_mean(feature_collection_index, feature_index)
        feature_collection_precision = get_collection_precision(feature_collection_index, feature_index)

        mean_addition = update.update_mean(self, feature_collection_mean)
        precision_addition, precision_interaction = update.update_precision(self, feature_collection_precision)

        feature_end_index = self.get_feature_end_index(feature_collection_index, feature_index)

        #Splice in the updated mean
        mean_start = self.mean_matrix[:, :feature_index]
        mean_end = self.mean_matrix[:, feature_end_index:]
        self.mean_matrix = np.hstack((mean_start, mean_addition, mean_end))


        #TODO: would probably be easier to just allocate one beeg tensor and copy stuff over, lel
        #TODO: Also probably doesn't belong here, but in some utils file or somethin
        #Splice in the updated precision
        #this is [t, s0, t, s0]
        precision_start = self.precision_tensor[:, :feature_index, :, :feature_index]
        #this is [t, s2, t, s2]
        precision_end = self.precision_tensor[:, feature_end_index:, :, feature_end_index:]
        #[t, s0, t, s2]
        precision_start_end = self.precision_tensor[:, :feature_index, :, feature_end_index:]
        #[t, s2, t, s0]
        precision_end_start = self.precision_tensor[:, feature_end_index:, :feature_index]

        #Shape [t, s1, t, s1]
        precision_middle = precision_addition

        #Shape [t, s1, t, s0 + s2]
        precision_middle_r = precision_interaction

        #Shape [t, s1, t, s0]
        precision_middle_r_zero = precision_interaction[:, :, :, :feature_index]

        #Shape [t, s1, t, s2]
        precision_middle_r_two = precision_interaction[:, :, :, feature_end_index:]

        #Shape [t, s0, t, s1]
        precision_middle_l_zero = np.transpose(precision_middle_r_zero, [0, 3, 2, 1])

        #Shape [t, s2, t, s1]
        precision_middle_l_two = np.transpose(precision_middle_r_two, [0, 3, 2, 1])

        #Shape [t, s0, t, s0 + s1 + s2]
        precision_first_row = np.concatenate((precision_start, precision_middle_l_zero, precision_start_end), axis=3)
        #Shape [t, s1, t, s0 + s1 + s2]
        precision_second_row = np.concatenate((precision_middle_r_zero, precision_middle, precision_middle_r_two), axis=3)
        #Shape [t, s2, t, s0 + s1 + s2]
        precision_third_row = np.concatenate((precision_end_start, precision_middle_l_two, precision_end), axis=3)

        #Shape [t, s0 + s1 + s2, t, s0 + s1 + s2]
        self.precision_tensor = np.concatenate((precision_first_row, precision_second_row, precision_third_row), axis=1)









