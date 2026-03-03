# https://www.tsc.uc3m.es/~fernando/bare_conf3.pdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
def get_kl_divergence(x1, x2):
    """
    The main function to calculate kl_divergence.
    @param  x1          experimental data points
    @param  x2          simulated data points
    @return divergence  KL divergence approximated using Eq. (4) of the paper
    """
    # Sort data and take unique values
    exp_x = sort_uniques(x1)
    sim_x = sort_uniques(x2)
    
    # Generate empirical cdfs
    x0 = np.min(exp_x) - 1 # Arbitrarily chosen based on description after Eq. (3)
    xn_1 = np.max(exp_x) + 1    
    exp_cdf = ecdf(exp_x) # has length x+2
    sim_cdf = ecdf(sim_x)
    
    # Calculate epsilon
    eps = np.min(np.concatenate((np.diff(exp_x), np.diff(sim_x))))/2
    
    # Calculate divergence
    exp_x_ext = insert_first_last(exp_x, x0, xn_1) # Used for interpolation
    sim_x_ext = insert_first_last(sim_x, x0, xn_1)
    divergence = divergence_eq(exp_x, exp_x_ext, exp_cdf, sim_x, sim_x_ext, sim_cdf, eps) - 1
    
    return divergence
    
def divergence_eq(x1, x1_ext, cdf1, x2, x2_ext, cdf2, eps):
    """
    Eq. (4) in the paper. Returns KL divergence estimate.
    @param  x1      Data points of reference function P(x)
    @param  x1_ext  x1 with x0 and xn+1 inserted
    @param  cdf1    cumulative distribution function of reference function
    @param  eps1    epsilon for reference function
    @param  x2      Data points for new function Q(x)
    ...
    @return KL divergence estimate
    """
    n = len(x1)
    delta_P = dP(x1, x1_ext, cdf1, eps)
    delta_Q = dP(x1, x2_ext, cdf2, eps) # Use the same x_i as P
    
    return np.sum(np.log(delta_P/delta_Q))/n

def dP(x_i, x, y, eps):
    '''
    Calculate Pc(x) by interpolating y(x_i) on the continuous cdf.
    Part of Eq. (4) in the paper
    
    @param  x_i     array of points to interpolate at
    @param  x       array of samples
    @param  y       empirical cdf of x
    @param  eps     = min(
    
    Give as input x=x_ext, y=cdf
    '''
    p_x_i = np.interp(x_i, x, y, left=0, right=1)
    p_x_i_minus_eps = np.interp(x_i - eps, x, y, left=0, right=1)
    return p_x_i - p_x_i_minus_eps

def ecdf(data_points, y0=0, yn1=1):
    """
    Generates an empirical cdf as described in Eq. (3)
    """
    x = np.sort(data_points)
    x = np.asarray(list(set(x))) # get unique values
    n = len(x)
    u = np.ones(n)
    y = (np.cumsum(u) - 0.5)/n
    
    # Insert end values
    y = insert_first_last(y, y0, yn1)
    
    return y

def sort_uniques(nparray):
    """
    Get unique values in array and then sort from smallest to largest
    """
    result = np.asarray(list(set(nparray))) # get unique values
    result = np.sort(nparray)
    return result
    
def insert_first_last(array, x0, xn_1):
    """
    Inserts x0 into the 0th position and xn_1 into the last position of array
    """
    array_ext = np.insert(array, 0, x0)
    array_ext = np.append(array_ext, xn_1)
    return array_ext
    
def plot_ecdf(exp_x, exp_x_ext, exp_cdf):
    # Generate continuous piece-wise linear function of cdf
    ynew = np.interp(exp_x, exp_x_ext, exp_cdf)
    plt.plot(exp_x, ynew, '-', label='linear interp')
    plt.plot(exp_x_ext, exp_cdf, 'o', label='data')
    plt.legend(loc='best')
    plt.show()
    
if __name__ == '__main__':
    main()
    
