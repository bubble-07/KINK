import numpy as np
import math
import scipy as sp
import scipy.spatial.distance

import matplotlib
import matplotlib.pyplot as plt

LINEAR_LAMBDA = 10.0
GAUSSIAN_LAMBDA = 10.0
GAUSSIAN_START_SCALE = 0.3
GAUSSIAN_SCALING = 2.0
NUM_SCALES = 3

NUM_TRAINING_POINTS = 100
NUM_TEST_POINTS = 100
NOISE = 0.0

#Given the passed data matrix [n x d],
#returns the n x n kernel matrix
#given by evaluating the gaussian kernel
#1/(\alpha sqrt(2) * sqrt(2 \pi)^d) * e^-||x - y||^2 / (4 \alpha^2)
def gaussian_kernel_matrix(alpha, X):
    n, d = X.shape
    
    dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, 'sqeuclidean'))

    exp_mat = np.exp((-0.25 / (alpha * alpha)) * dist_mat)

    scaling_factor = (1.0 / (alpha * math.sqrt(2) * ((math.sqrt(2) * math.pi) ** d)))

    return exp_mat * scaling_factor

def gaussian_kernelize(alpha, X, v):
    n, d = X.shape

    sq_dists = X - v
    sq_dists = sq_dists * sq_dists
    sq_dists = np.sum(sq_dists, axis=1)

    exp_vec = np.exp((-0.25 / (alpha * alpha)) * sq_dists)

    scaling_factor = (1.0 / (alpha * math.sqrt(2) * ((math.sqrt(2) * math.pi) ** d)))

    return exp_vec * scaling_factor

def linear_kernelize(X, v):
    return np.matmul(X, v)

def linear_kernel_matrix(X):
    result = np.matmul(X, np.transpose(X))
    return result

def combine_lambda(lambda_one, lambda_two):
    return math.sqrt(lambda_one * lambda_two)

#Gets the parameter alpha to use for computing
#interaction terms between different kernel spaces
def get_interaction_alpha(alpha_one, alpha_two):
    alpha_one_sq = alpha_one * alpha_one
    alpha_two_sq = alpha_two * alpha_two
    alpha_sum_sq = alpha_one_sq + alpha_two_sq
    return math.sqrt(alpha_sum_sq * 0.5)

def gaussian_interaction_kernel_matrix(alpha_one, alpha_two, X):
    alpha_prime = get_interaction_alpha(alpha_one, alpha_two)
    return gaussian_kernel_matrix(alpha_prime, X)

class LinearKernelSpec(object):
    def __init__(self, reg_lambda):
        self.reg_lambda = reg_lambda

    def interaction_prior_precision(self, other_spec, X):
        n, d = X.shape
        combined_lambda = combine_lambda(self.reg_lambda, other_spec.reg_lambda)
        return n * combined_lambda * self.interaction_kernel_matrix(other_spec, X)
    
    def interaction_kernel_matrix(self, other_spec, X):
        n, d = X.shape
        if (isinstance(other_spec, LinearKernelSpec)):
            return linear_kernel_matrix(X)
        else:
            return linear_kernel_matrix(X)

    def kernelize(self, X, v):
        return linear_kernelize(X, v)
    def kernel_matrix(self, X):
        return linear_kernel_matrix(X)

class GaussianKernelSpec(object):
    def __init__(self, reg_lambda, alpha):
        self.reg_lambda = reg_lambda
        self.alpha = alpha

    def interaction_kernel_matrix(self, other_spec, X):
        n, d = X.shape
        if (isinstance(other_spec, LinearKernelSpec)):
            return other_spec.interaction_kernel_matrix(self, X)
        else:
            return gaussian_interaction_kernel_matrix(self.alpha, other_spec.alpha, X)

    def interaction_prior_precision(self, other_spec, X):
        n, d = X.shape
        combined_lambda = combine_lambda(self.reg_lambda, other_spec.reg_lambda)
        return n * combined_lambda * self.interaction_kernel_matrix(other_spec, X)

    def kernelize(self, X, v):
        return gaussian_kernelize(self.alpha, X, v)
    def kernel_matrix(self, X):
        return gaussian_kernel_matrix(self.alpha, X)

class MultiKernelSpec(object):
    def __init__(self, specs):
        self.specs = specs

    def prior_precision(self, X):
        result_block_mat = []
        for spec_one in self.specs:
            result_block_line = []
            for spec_two in self.specs:
                block = spec_one.interaction_prior_precision(spec_two, X)
                result_block_line.append(block)
            result_block_mat.append(result_block_line)
        result = np.block(result_block_mat)
        return result
    
    def kernelize(self, X, v):
        kernelized = []  
        for spec in self.specs:
            kernelized.append(spec.kernelize(X, v))
        return np.concatenate(kernelized)

    def interaction_kernel_matrix(self, X):
        result_block_mat = []
        for spec_one in self.specs:
            result_block_line = []
            for spec_two in self.specs:
                block = spec_one.interaction_kernel_matrix(spec_two, X)
                result_block_line.append(block)
            result_block_mat.append(result_block_line)
        result = np.block(result_block_mat)
        return result

    def kernel_matrix(self, X):
        blocks = []
        for spec in self.specs:
            K = spec.kernel_matrix(X)
            blocks.append(K)
        blocks = np.hstack(blocks)
        return blocks

    def fit_model_parameters(self, X, y):
        K = self.kernel_matrix(X)
        print K
        KT = np.transpose(K)
        KTK = np.matmul(KT, K)
        prior_precision = self.prior_precision(X)
        print prior_precision
        precision = KTK + prior_precision
        covar = np.linalg.pinv(precision)
        KTy = np.matmul(KT, y)
        return np.matmul(covar, KTy)
    
    def use_model_parameters(self, X, u, v):
        kernelized_v = self.kernelize(X, v)
        return np.dot(u, kernelized_v)

specs = []
current_lambda = GAUSSIAN_LAMBDA
current_bandwidth = GAUSSIAN_START_SCALE
for i in range(NUM_SCALES):
    spec = GaussianKernelSpec(current_lambda, current_bandwidth)
    current_bandwidth /= GAUSSIAN_SCALING
    current_lambda *= GAUSSIAN_SCALING
    specs.append(spec)

lin_spec = LinearKernelSpec(LINEAR_LAMBDA)
specs.append(lin_spec)

multi_kernel_spec = MultiKernelSpec(specs)



def func(x):
    #return x * x
    #return math.e ** (-10 * x * x)
    return x

train_inputs = np.random.uniform(-1.0, 1.0, NUM_TRAINING_POINTS)
train_outputs = func(train_inputs) + np.random.normal(scale=NOISE, size=NUM_TRAINING_POINTS)
train_inputs = train_inputs.reshape([-1, 1])

model_params = multi_kernel_spec.fit_model_parameters(train_inputs, train_outputs)

test_inputs = np.arange(-2.0, 2.0, 4.0 / NUM_TEST_POINTS)

test_outputs = []
for i in range(NUM_TEST_POINTS):
    test_input = test_inputs[i].reshape(1)
    test_output = multi_kernel_spec.use_model_parameters(train_inputs, model_params, test_input)
    test_outputs.append(test_output)
test_outputs = np.array(test_outputs)

fig, ax = plt.subplots()
ax.plot(test_inputs, test_outputs)
ax.scatter(train_inputs, train_outputs)
ax.grid()

plt.show()


