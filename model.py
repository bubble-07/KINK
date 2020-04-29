import numpy as np

class Model(object):

    def __init__(self, feature_collections):
        self.feature_collections = feature_collections
        self.num_in_dims = 0
        self.num_out_dims = 0
        self.num_data_points = 0

    def decompose_mean(self):
        """
        Decomposes the mean matrix into (t x s_i)-size blocks
        by feature collection
        """
        start_ind = 0
        result = []
        for feature_collection in self.feature_collections:
            end_ind = start_ind + feature_collection.get_features_dimension()
            result.append(self.mean_matrix[:, start_ind:end_ind])
            start_ind = end_ind
        return result

    def decompose_precision(self):
        """
        Decomposes the precision matrix into (t x s_i) x (t x s_j)-size
        blocks by feature collection
        """
        start_i = 0
        result = []
        for collection_i in self.feature_collections:
            end_i = start_i + collection_i.get_features_dimension()

            start_j = 0
            row = []
            for collection_j in self.feature_collections:
                end_j = start_j + collection_j.get_features_dimension()
                row.append(self.precision_tensor[:, start_i:end_i, :, start_j:end_j])
                start_j = end_j
            result.append(row)
            start_i = end_i
        return result

    def set_num_in_dims(self, num_in_dims):
        self.num_in_dims = num_in_dims
        self.update_feature_collections() 

    def set_num_data_points(self, num_data_points):
        self.num_data_points = num_data_points
        self.update_feature_collections()

    def update_feature_collections():
        for i in range(len(self.feature_collections)):
            feature_collection = self.feature_collections[i]

            maybe_model_update = feature_collection.set_dimension(self.num_in_dims)
            if (maybe_model_update is not None):
                self.apply_model_update(maybe_model_update, i)

            maybe_model_update = feature_collection.set_num_data_points(self.num_data_points)
            if (maybe_model_update is not None):
                self.apply_model_update(maybe_model_update, i)


    def set_num_out_dims(self, num_out_dims):
        """
        Adjusts the number of output dimensions on this model
        to the prescribed number
        """
        means = self.decompose_mean()
        precisions = self.decompose_precision()

        #Adjust means
        new_means = []
        for i in range(len(self.feature_collections)):
            feature_collection = self.feature_collections[i]
            prior_mean = means[i]
            new_means.append(feature_collection.adjust_mean_output_dims(prior_mean, num_out_dims))
        self.mean_matrix = np.concatenate(new_means, axis=1)

        #Adjust precisions
        new_precisions = []
        for i in range(len(self.feature_collections)):
            collection_i = self.feature_collections[i]
            row = []
            for j in range(len(self.feature_collections)):
                collection_j = self.feature_collections[j]
                prior_precision = precisions[i][j]
                new_precision = collection_i.adjust_precision_output_dims(collection_j, prior_precision, num_out_dims)
                row.append(new_precision)
            concat_row = np.concatenate(row, axis=3)
            new_precisions.append(concat_row)

        self.precision_tensor = np.concatenate(new_precisions, axis=1)

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

    def apply_model_update(self, update, collection_index):
        """
        Helper function to apply the given ModelUpdate
        originating from an update on the passed collection_index
        """
        means = self.decompose_mean()
        precisions = self.decompose_precision()

        old_mean = means[collection_index]
        new_mean = update.update_mean(self, old_mean)
        means[collection_index] = new_mean
        self.mean_matrix = np.concatenate(means, axis=1)

        for other_index in range(len(self.feature_collections)):
            other_collection = self.feature_collections[other_index]
            old_precision = precisions[collection_index][other_index]  
            new_precision = update.update_precision(self, other_collection, old_precision)

            #Assign to the index we got it from
            precisions[collection_index][other_index] = new_precision
            #But also transpose and assign to the symmetric index
            new_precision_t = np.transpose(new_precision, axes=[0, 3, 2, 1])
            precisions[other_index][collection_index] = new_precision_t

        precisions_concat = []
        for row in precisions:
            precisions_concat.append(np.concatenate(row, axis=3))
        self.precision_tensor = np.concatenate(precisions_concat, axis=1)

