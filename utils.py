import numpy as np

def copy_from_into_4D(tensor_one, tensor_two):
    a, b, c, d = tensor_one.shape
    d, e, f, g = tensor_two.shape
    h, i, j, k = (min(a, d), min(b, e), min(c, f), min(d, g))
    tensor_two[:h, :i, :j, :k] = tensor_one[:h, :i, :j, :k]

#Computes the square root of the "closest" positive semi-definite matrix
#to a given hermitian matrix
def sqrtm(X):
    D, W = np.linalg.eigh(X)
    D = np.maximum(0, D) #All must be non-negative due to the PSD assumption
    sqrt_D = np.sqrt(D)
    #The result comes from scaling the columns appropriately
    result = np.multiply(W, sqrt_D.reshape((-1, 1)))
    return result
