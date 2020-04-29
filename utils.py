import numpy as np

def copy_from_into_4D(tensor_one, tensor_two):
    a, b, c, d = tensor_one.shape
    d, e, f, g = tensor_two.shape
    h, i, j, k = np.min([a, b, c, d], [d, e, f, g])
    tensor_two[:h, :i, :j, :k] = tensor_one[:h, :i, :j, :k]
