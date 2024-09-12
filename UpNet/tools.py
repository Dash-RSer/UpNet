import numpy as np
import scipy.stats as stats
import numba
import warnings
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import shelve
warnings.filterwarnings('ignore')

def truncated_normal(mean, sigma, lower, upper, n_samples):
    """
    Pass
    """
    x = stats.truncnorm(
        (lower - mean) / sigma, 
        (upper - mean) / sigma,
        loc=mean, scale=sigma)
    samples = np.array(x.rvs([n_samples]), dtype = np.float32)
    return samples

def concatenate_array(array, N_observation):
    N = array.shape[0]
    concatenated_array = np.zeros(N_observation, dtype= np.float32)
    index = 0
    for i in range(N):
        sub_array = array[i]
        n_elements = sub_array.shape[0]
        concatenated_array[index:index+n_elements] = sub_array
        index = index+n_elements
    return concatenated_array


def build_covariance(n_obversation, noise_std_dev):
    covariance_matrix = \
        np.zeros((n_obversation, n_obversation), dtype = np.float32)
    for i in range(n_obversation):
        covariance_matrix[i, i] = noise_std_dev**2
    return covariance_matrix

def build_covariance_ratio(y_real, ratio = 0.1):
    n_ob = len(y_real)
    covariance_matrix = np.zeros((n_ob, n_ob), dtype = np.float32)
    for i in range(n_ob):
        covariance_matrix[i,i] = (y_real[i]*ratio)**2
    return covariance_matrix

def noise_generator(dimensions, sigma, cut):
    return truncated_normal(0, sigma, -cut, cut, dimensions)

def noise_generator_ratio(observation, cut, n_obversation, ratio = 0.05):
    noise = np.zeros(n_obversation, dtype = np.float32)
    for i in range(n_obversation):
        noise[i] = truncated_normal(0, observation[i]*ratio, -cut, cut, 1)
    return noise

def samples_generator(parameters, n):
    """
    parameters: ['g', mean, sigma, lower, upper] for truncated gaussian density.
                ['u', lower_bound, upper_bound] for uniform density.
    """
    prior_type = parameters[0]
    if prior_type == 'g' or prior_type == 'tg':
        mean = parameters[1]
        sigma = parameters[2]
        lower = parameters[3]
        upper = parameters[4]
        return np.float32(truncated_normal(mean, sigma, lower, upper, n))
    elif prior_type == 'u':
        lower = parameters[1]
        upper = parameters[2]
        return np.float32(np.random.uniform(lower, upper, n))
    else:
        raise Exception("undefined prior type")

def batch_samples_generator(paras_list, n_samples):
    n_paras = len(paras_list)
    samples = np.zeros((n_samples, n_paras), dtype = np.float32)
    for i in range(n_paras):
        samples[:, i] = samples_generator(paras_list[i], n_samples)
    return samples

if __name__ == "__main__":
    pass