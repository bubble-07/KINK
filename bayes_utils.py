import numpy as np
import utils

def update_normal_inverse_gamma(params, data_tuple):
    """
    Given a collection of parameters of the format
    (u, precision_u, /\, sigma, a, b), yield the updated parameters due to
    adding the single data point given by data_tuple in the format
    (in_vec : s, out_vec : t, out_precision : t x t)
    """
    mean, precision_u, precision, sigma, a, b = params
    in_vec, out_vec, out_precision = data_tuple

    t, s = mean.shape

    U = utils.sqrtm(out_precision)

    precision_contrib = np.einsum('ac,b,d->abcd', out_precision, in_vec, in_vec)

    result_precision = precision + precision_contrib


    #We're using the woodbury matrix identity here to compute
    #the updated covariance [inverse of precision]

    #Compute [sigma (x o U)] : (t x s) x t
    sigma_x_U = np.einsum('abcd,d,ce->abe', sigma, in_vec, U)

    #Compute [(x^T o U) sigma] : t x (t x s)
    x_T_U_sigma = np.einsum('a,bc,cade->bde', in_vec, U, sigma)

    #Compute [(x^T o U) sigma (x o U)] : t x t
    x_T_U_sigma_x_U = np.einsum('abc,c,bd->ad', x_T_U_sigma, in_vec, U)

    #Compute Z = I + (prev) : t x t
    Z = x_T_U_sigma_x_U + np.eye(t)
    Z_inv = np.pinv(expression_to_invert)

    #Compute sigma_diff = [sigma_x_U  Z_inv  x_T_U_sigma] : (t x s) x (t x s)
    sigma_diff = np.einsum('abe,ef,fcd', sigma_x_U, Z_inv, x_T_U_sigma)

    result_sigma = sigma - sigma_diff

    
    #Compute the updated /\ * u

    #Compute x o (out_precision y) : (t x s)
    x_out_precision_y = np.einsum('sab,b->as', in_vec, out_precision, out_vec)

    result_precision_u = precision_u + x_out_precision_y
    

    #Compute the updated u
    result_mean = np.einsum('abcd,cd->ab', result_sigma, result_precision_u)


    #Compute the updated a
    result_a = a + (t / 2.0)


    #Compute y^T out_precision y
    y_T_out_precision_y = np.einsum('x,xy,y->', out_vec, out_precision, out_vec)

    #Compute u_0 precision u_0
    u_precision_u_zero = np.einsum('ab,ab->', mean, precision_u)

    #Compute u_n precision u_n
    u_precision_u_n = np.einsum('ab,ab->', result_mean, result_precision_u)

    result_b = b + y_T_out_precision_y + 0.5 * (u_precision_u_zero - u_precision_u_n)


    return (result_mean, result_precision_u, result_precision, result_sigma, result_a, result_b)


def combine_normal_inverse_gammas(params_one, params_two):
    """
    Given two collections of parameters of the format
    (u, /\, sigma, a, b), yield the normal-inverse-gamma sum
    [as in https://projecteuclid.org/download/pdfview_1/euclid.ba/1510110046]
    of the two parameter tuples as a new parameter tuple

    Here, u is of dimensions t x s, /\ is of dimensions ((t x s) x (t x s)) and so is sigma
    a is a scalar, and b is a scalar
    """
    mean_one, precision_one, precision_u_one, sigma_one, a_one, b_one = params_one
    mean_two, precision_two, precision_u_two, sigma_two, a_two, b_two = params_two

    t, s = mean_one.shape

    precision_out = precision_one + precision_two


    precision_out_mat = precision_out.reshape((t * s, t * s))
    precision_out_mat_inv = np.pinv(precision_out_mat)
    sigma_out = precision_out_mat_inv.reshape((t, s, t, s))

    l_one_u_one = np.einsum('abcd,cd->ab', precision_one, mean_one)
    l_two_u_two = np.einsum('abcd,cd->ab', precision_two, mean_two)

    precision_u_out = l_one_u_one + l_two_u_two

    mean_out = np.einsum('abcd,cd->ab', sigma_out, precision_u_out)


    a_out = a_one + a_two + (t * s) / 2.0


    mean_one_diff = mean_one - mean_out
    mean_two_diff = mean_two - mean_out

    u_diff_l_u_diff_one = np.einsum('ab,abcd,cd->', mean_one_diff, precision_one, mean_one_diff)
    u_diff_l_u_diff_two = np.einsum('ab,abcd,cd->', mean_two_diff, precision_two, mean_two_diff)
    u_diff_l_u_diff_i = u_diff_l_u_diff_one + u_diff_l_u_diff_two

    b_out = b_one + b_two + 0.5 * u_diff_l_u_diff_i


    return (mean_out, precision_u_out, precision_out, sigma_out, a_out, b_out)
    
