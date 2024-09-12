import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from tools import noise_generator_ratio, build_covariance_ratio, build_covariance
import pymc as pm
import pytensor
import pytensor.tensor as pt
from ProSAIL.RTMS import ProSAIL

def retrieval_model(parameters, design):
    LAI, Cab, N, Cw, Cm, ALA, psoil = parameters
    tts, tto, psi, bands = design
    ref = ProSAIL(N, Cab, 8, Cw, Cm, LAI, ALA, 0.3, 0.8, psoil, tts, tto, psi, bands)
    return ref

def log_likelihood(parameters_all, y, design, cov_func):
    if cov_func is None:
        y_simu = retrieval_model(parameters_all, design)
    else:
        y_simu = cov_func(retrieval_model(parameters_all, design))
    n_observations = y_simu.shape[0]
    covariance_matrix = build_covariance_ratio(y_simu, 0.05) + build_covariance(n_observations, 0.01)
    log_likelihood_sum = 0
    constant_sum = -(n_observations/2)*np.log(2*np.pi)
    kernel_sums = 0
    for j in range(n_observations):
        kernel_sums = kernel_sums - (1/(2*covariance_matrix[j, j]))*((y[j] - y_simu[j])**2)
        constant_sum = constant_sum - np.log(covariance_matrix[j, j])
    log_likelihood_sum = log_likelihood_sum + kernel_sums + constant_sum
    return log_likelihood_sum

class LogLike(pt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [pt.dvector]  # expects a matrix of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, y, n_paras, design, cov_func):

        # add inputs as class attributes
        self.likelihood = loglike
        self.y = y
        self.n_paras = n_paras
        self.design = design
        self.cov_func = cov_func

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (parameters, ) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(parameters, self.y, self.design, self.cov_func)
        outputs[0][0] = np.array(logl)  # output the log-likelihood
        return outputs

def MCMC_inference(ys, parameter_index = 0, cov_func = None):
    bands = [482, 561, 655, 865, 1609, 2201]
    # bands = [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]
    # bands = np.arange(400, 2501, 1)
    logl = LogLike(log_likelihood, ys[:-1], 7, [ys[-1], 0, 0, bands], cov_func)
    with pm.Model():
        # LAI = pm.Uniform("LAI", lower = 0, upper = 7)
        LAI = pm.TruncatedNormal('LAI', mu = 3, sigma = 2, lower = 0, upper = 7)
        Cab = pm.TruncatedNormal('Cab', mu = 35, sigma = 30, lower = 5, upper = 75)
        N = pm.Uniform("Ns", lower = 1.3, upper = 2.5)
        Cw= pm.TruncatedNormal('Cw', mu = 0.02, sigma = 0.01, lower = 0.002, upper = 0.05)
        Cm = pm.TruncatedNormal('Cm', mu = 0.005, sigma = 0.001, lower = 0.001, upper = 0.03)
        ALA = pm.Uniform("ALA",lower = 40, upper = 70)
        psoil = pm.Uniform("psoil",lower = 0, upper = 1)

        parameters = pt.as_tensor_variable([LAI, Cab, N, Cw, Cm, ALA, psoil])
        pm.Potential("likelihood", logl(parameters))
        idata_mh = pm.sample(500, tune = 100, chains = 1, step=pm.Metropolis())

    stats = az.summary(idata_mh)
    return np.array([stats['mean'][0], stats['sd'][0], stats['mean'][1], stats['sd'][1]])

if __name__ == "__main__":    
    # LAI, Cab, N, Cm, ALA, Hspot
    paras = [1.5, 40, 1.5, 0.01, 0.005, 65, 0.2]
    bands = [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]
    
    y = retrieval_model(paras, [30, 0, 0, bands])
    mean, std = MCMC_inference(y)
    print(mean, std)
    pass
    



