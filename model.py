import numpy as np
import bayes_utils

class NormalInverseGamma(object):
    def __init__(self, mean, precision, a, b):
        bayes_tuple = bayes_utils.expand_params((mean, precision, a, b))
        self.set_bayes_tuple(bayes_tuple)

    def get_bayes_tuple(self):
        return (self.mean, self.precision_u, self.precision, self.sigma, self.a, self.b)

    def set_bayes_tuple(self, bayes_tuple):
        self.mean, self.precision_u, self.precision, self.sigma, self.a, self.b = bayes_tuple

    def get_model_tuple(self):
        return (self.mean, self.precision)

    def set_model_tuple(self, model_tuple):
        mean, precision = model_tuple
        bayes_tuple = bayes_utils.expand_params((mean, precision, self.a, self.b))
        self.set_bayes_tuple(bayes_tuple)

    def add_data(self, data_tuple):
        """
        Adds data from a data point [in_point, out_point, out_precision]
        """
        bayes_tuple = self.get_bayes_tuple()
        result_tuple = bayes_utils.update_normal_inverse_gamma(bayes_tuple, data_tuple, downdate=False)
        self.set_bayes_tuple(result_tuple)
        
    def subtract_data(self, data_tuple):
        bayes_tuple = self.get_bayes_tuple()
        result_tuple = bayes_utils.update_normal_inverse_gamma(bayes_tuple, data_tuple, downdate=True)
        self.set_bayes_tuple(result_tuple)
    
    def add_other(self, other): 
        """
        Adds the data from other [another normal inverse gamma]
        to this one to update this instance
        """
        bayes_tuple = self.get_bayes_tuple()
        other_bayes_tuple = other.get_bayes_tuple()
        result_bayes_tuple = bayes_utils.combine_normal_inverse_gammas(bayes_tuple, other_bayes_tuple)
        self.set_bayes_tuple(result_bayes_tuple)

    def subtract_other(self, other):
        """
        Subtracts the data from other [another normal inverse gamma]
        from this one to update this instance 
        """
        bayes_tuple = self.get_bayes_tuple()
        other_bayes_tuple = other.get_bayes_tuple()
        neg_bayes_tuple = bayes_utils.invert_normal_inverse_gamma(other_bayes_tuple)
        result_bayes_tuple = bayes_utils.combine_normal_inverse_gammas(bayes_tuple, neg_bayes_tuple)
        self.set_bayes_tuple(result_bayes_tuple)


class ModelSpace(object):

    def __init__(self, feature_collections):
        self.feature_collections = feature_collections
        self.num_in_dims = 0
        self.num_out_dims = 0

        #Dictionary from arbitrary keys to the number of data points
        #per model
        self.num_data_points = {}

        #The current maximum number of data points across all in num_data_points
        self.max_num_data_points = 0

        #Dictionary from arbitrary keys to parameters for models
        #in this model space, as tuples (mean_matrix, precision_tensor)
        self.models = {}

        #Dictionary from the same kind of keys used for self.models
        #to keyed dictionaries of updates which have been applied
        #to the respective models [so that they may be easily undone/recalled] 
        self.updates = {}

    def add_update(self, model_key, update_key, update):
        """
        Adds an update with a given key
        """
        self.models[model_key].add_other(update)
        self.updates[model_key][update_key] = update

    def remove_update(self, model_key, update_key):
        """
        Removes an update with a given key
        """
        update = self.updates[model_key][update_key]
        self.models[model_key].subtract_other(update)
        del self.updates[model_key][update_key]

    def featureify(self, x):
        f = np.zeros(self.get_total_feature_dimensions())
        start_ind = 0
        for i in range(len(self.feature_collections)):
            feature_collection = self.feature_collections[i]
            end_ind = start_ind + feature_collection.get_features_dimension()
            v = feature_collection.get_features(x)
            f[start_ind:end_ind] = v
        return f

    def add_datapoint(self, model_key, x, y, out_precision):
        """
        Given a datapoint [tuple with sizes num_in, num_out, num_out x num_out]
        update the given model
        """
        data_tuple = (self.featureify(x), y, out_precision)
        self.models[model_key].add_data(data_tuple)

    def remove_datapoint(self, model_key, x, y, out_precision):
        data_tuple = (self.featureify(x), y, out_precision)
        self.models[model_key].subtract_data(data_tuple)

    def remove_model(self, model_key):
        """
        Removes a model with the given model key
        """
        del self.models[model_key]
        del self.updates[model_key]

    def add_model(self, model_key):
        """
        Adds a new model with the given model key
        """
        t = self.num_out_dims
        s = self.get_total_feature_dimensions()
        mean = np.zeros((t, s))
        precision = np.zeros((t, s, t, s))
        a = -0.5 * s * t
        b = 0

        start_ind = 0
        for i in range(len(self.feature_collections)):
            feature_collection = self.feature_collections[i]
            end_ind = start_ind + feature_collection.get_features_dimension() 
            block = feature_collection.blank_precision(feature_collection, t)
            precision[:, start_ind:end_ind, :, start_ind:end_ind] = block
            start_ind = end_ind

        model = NormalInverseGamma(mean, precision, a, b)
        self.models[model_key] = model
        self.updates[model_key] = {}

    def decompose_mean(self, mean_matrix):
        """
        Decomposes a mean matrix into (t x s_i)-size blocks
        by feature collection
        """
        start_ind = 0
        result = []
        for feature_collection in self.feature_collections:
            end_ind = start_ind + feature_collection.get_features_dimension()
            result.append(mean_matrix[:, start_ind:end_ind])
            start_ind = end_ind
        return result

    def decompose_precision(self, precision_tensor):
        """
        Decomposes a precision tensor into (t x s_i) x (t x s_j)-size
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
                row.append(precision_tensor[:, start_i:end_i, :, start_j:end_j])
                start_j = end_j
            result.append(row)
            start_i = end_i
        return result

    def set_num_in_dims(self, num_in_dims):
        self.num_in_dims = num_in_dims
        self.update_feature_collections() 

    def set_num_data_points(self, model_key, num_data_points):
        self.num_data_points[model_key] = num_data_points
        self.max_num_data_points = max(self.max_num_data_points, num_data_points)
        self.update_feature_collections()

    def get_total_feature_dimensions(self):
        result = 0
        for feature_collection in self.feature_collections:
            result += feature_collection.get_features_dimension()
        return result

    def update_feature_collections():
        #Before doing anything, record all of the current dimensions of the feature collections
        #We will use this to perform a bayesian update simulating an adjusted prior
        #since the prior is scaled according to s, where s is the number of features
        old_dims = []
        for feature_collection in self.feature_collections:
            old_dims.append(feature_collection.get_features_dimension())

        anything_changed = False
        for i in range(len(self.feature_collections)):
            feature_collection = self.feature_collections[i]

            maybe_model_update = feature_collection.set_dimension(self.num_in_dims)
            if (maybe_model_update is not None):
                self.apply_model_update(maybe_model_update, i)
                anything_changed = True

            maybe_model_update = feature_collection.set_num_data_points(self.max_num_data_points)
            if (maybe_model_update is not None):
                self.apply_model_update(maybe_model_update, i)
                anything_changed = True

        if anything_changed:
            #Here, we need to still perform a bayesian update to the precision matrix
            #All of the elements outside the ranges of the original tensor slices
            #were properly updated to reflect the new dimensions, however, we need
            #to go back and apply the update diff within the range
            t = self.num_out_dims
            s = self.get_total_feature_dimensions()
            mean_delta = np.zeros((t, s))
            precision_delta = np.zeros((t, s, t, s))
            b_delta = 0
            a_delta = -0.5 * (s * t)

            #Fill in precision delta
            feat_start_ind = 0
            for i in range(len(self.feature_collections)):
                feature_collection = self.feature_collections[i]
                orig_size = old_dims[i]
                new_size = feature_collection.get_total_feature_dimensions()
                feat_end_ind = feat_start_ind + orig_size

                diff_size = new_size - orig_size
                block = np.kron(np.eye(t), np.eye(orig_size))
                block *= feature_collection.reg_factor * diff_size
                block = block.reshape(block, (t, orig_size, t, orig_size))
                precision_delta[:, feat_start_ind:feat_end_ind, :, feat_start_ind:feat_end_ind] = block

                feat_start_ind += orig_size

            #Now that it's constructed, apply it to all of the models
            #but don't add it as an update
            update = NormalInverseGamma(mean_delta, precision_delta, a_delta, b_delta)
            for model_key in self.models:
                self.models[model_key].add_other(update)


    def set_num_out_dims(self, num_out_dims):
        """
        Adjusts the number of output dimensions on this model space
        to the prescribed number
        """
        for model_key in self.models:
            model_tuple = self.models[model_key].get_model_tuple()
            model_tuple = set_num_out_dims_helper(num_out_dims, model_tuple, update=False)
            self.models[model_key].set_model_tuple(model_tuple)

        for model_key in self.models:
            for update_key in self.updates[model_key]:
                update_tuple = self.updates[model_key][update_key]
                update_tuple = set_num_out_dims_helper(num_out_dims, update_tuple, update=True)
                self.updates[model_key][update_key].set_model_tuple(update_tuple)
                #TODO: Do we need to update other things here, like a/b?

        self.num_out_dims = num_out_dims

    def set_num_out_dims_helper(self, num_out_dims, model_tuple, update=False):
        mean_matrix, precision_tensor = model_tuple

        means = self.decompose_mean(mean_matrix)
        precisions = self.decompose_precision(precision_tensor)

        #Adjust means
        new_means = []
        for i in range(len(self.feature_collections)):
            feature_collection = self.feature_collections[i]
            prior_mean = means[i]
            new_means.append(feature_collection.adjust_mean_output_dims(prior_mean, num_out_dims, update))
        mean_matrix = np.concatenate(new_means, axis=1)

        #Adjust precisions
        new_precisions = []
        for i in range(len(self.feature_collections)):
            collection_i = self.feature_collections[i]
            row = []
            for j in range(len(self.feature_collections)):
                collection_j = self.feature_collections[j]
                prior_precision = precisions[i][j]
                new_precision = collection_i.adjust_precision_output_dims(collection_j, prior_precision, num_out_dims, update)
                row.append(new_precision)
            concat_row = np.concatenate(row, axis=3)
            new_precisions.append(concat_row)

        precision_tensor = np.concatenate(new_precisions, axis=1)

        return (mean_matrix, precision_tensor)

    def apply_model_update(self, update, collection_index):
        """
        Helper function to apply a model update stemming from
        a particular feature collection on this model space
        """
        for model_key in self.models:
            model_tuple = self.models[model_key].get_model_tuple()
            model_tuple = self.apply_model_update_helper(update, collection_index, update=False)
            self.models[model_key].set_model_tuple(model_tuple)

        for model_key in self.models:
            for update_key in self.updates[model_key]:
                update_tuple = self.updates[model_key][update_key].get_model_tuple()
                update_tuple = self.apply_model_update_helper(update, collection_index, update=True)
                self.updates[model_key][update_key].set_model_tuple(update_tuple)
                #TODO: Are there other things we need to do here [like updating a/b?]

        
    def apply_model_update_helper(self, update, collection_index, model_tuple, update=False):
        """
        Helper function to apply the given ModelUpdate
        originating from an update on the passed collection_index
        to the passed model params
        """
        mean_matrix, precision_tensor = model_tuple

        means = self.decompose_mean(mean_matrix)
        precisions = self.decompose_precision(precision_tensor)

        old_mean = means[collection_index]
        new_mean = update.update_mean(self, old_mean, update)
        means[collection_index] = new_mean
        mean_matrix = np.concatenate(means, axis=1)

        for other_index in range(len(self.feature_collections)):
            other_collection = self.feature_collections[other_index]
            old_precision = precisions[collection_index][other_index]  
            new_precision = update.update_precision(self, other_collection, old_precision, update)

            #Assign to the index we got it from
            precisions[collection_index][other_index] = new_precision
            #But also transpose and assign to the symmetric index
            new_precision_t = np.transpose(new_precision, axes=[0, 3, 2, 1])
            precisions[other_index][collection_index] = new_precision_t

        precisions_concat = []
        for row in precisions:
            precisions_concat.append(np.concatenate(row, axis=3))
        precision_tensor = np.concatenate(precisions_concat, axis=1)

        return (mean_matrix, precision_tensor)

