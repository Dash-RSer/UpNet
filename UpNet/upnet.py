import numpy as np
import matplotlib.pyplot as plt
from tools import samples_generator, noise_generator_ratio, noise_generator, batch_samples_generator
from model import model_generator
from NN_model import Retriever
from MCMC import MCMC_inference
import warnings
import os
warnings.filterwarnings('ignore')
import time
from SRFCovolution import build_srf_covolution_func

retrieval_model = model_generator(['free', 'free', 8,  'free', 'free', 'free', 'free', 0.3, 'free'])

def test_nn(n_samples, 
            spectral = [482, 561, 655, 865, 1609, 2201], 
            weights_dir = 'weights\\landsat_8\\lai_cw',
            cov_function = None):
    
    n_observations = len(spectral)
    if os.path.exists(weights_dir) is False :
        Exception("No such weights dir.")
    mean_weight_path = os.path.join(weights_dir, 'mean_weight')
    mean_sc_path = os.path.join(weights_dir, 'mean_sc')
    var_weight_path = os.path.join(weights_dir, 'var_weight')
    var_sc_path = os.path.join(weights_dir, 'var_sc')

    config_list = [['g', 3, 2, 0, 7], ['g', 35, 30, 5, 75], ['u', 1.3, 2.5], ['g', 0.02, 0.01, 0.002, 0.05], 
                   ['g', 0.005, 0.001, 0.001, 0.03],['u', 40, 70], ['u', 0, 1]]
    samples = batch_samples_generator(config_list, n_samples)
    tts_samples = samples_generator(['u', 40, 70], n_samples)
    ys = np.zeros((samples.shape[0], n_observations + 1), dtype = np.float32)
    if cov_function is None :
        for i in range(n_samples):
            cur_y = retrieval_model(samples[i, :], [tts_samples[i], 0, 0, spectral])
            ys[i, :-1] = cur_y + noise_generator_ratio(cur_y, 0.12, n_observations, 0.04) + noise_generator(n_observations, 0.01, 0.03)
    else:
        for i in range(n_samples):
            cur_y = cov_function(retrieval_model(samples[i, :], [tts_samples[i], 0, 0, np.arange(400, 2501, 1)]))
            ys[i, :-1] = cur_y + noise_generator_ratio(cur_y, 0.12, n_observations, 0.04) + noise_generator(n_observations, 0.01, 0.03)
    ys[:, n_observations] = tts_samples

    retriever_mean = Retriever()
    retriever_mean.load_model(mean_weight_path, mean_sc_path)
    
    y_pred_mean = retriever_mean.predict(ys)
    retriever_var = Retriever(variance = True)
    retriever_var.load_model(var_weight_path, var_sc_path)
    y_pred_var = retriever_var.predict(ys)
    # remembter to change to std
    mcmc_pred_lais = []
    mcmc_pred_stds = []
    start = time.time()
    for i in range(n_samples):
        mean, std = MCMC_inference(ys[i, :], parameter_index = 0, cov_func = cov_function)
        mcmc_pred_lais.append(mean)
        mcmc_pred_stds.append(std)
    end = time.time()
    print('MCMC time:', end - start)
    print(".")
    
    # landsat8 lai : 11932.27 s
    # landsat8 cab: 11654.67s
    # sentinel-2 cab: 10067.15s
    # sentinel-2 lai: 11670.26s

def data_generator(n_samples, spectral = None, cov_function = None):
    """
    Generate the training dataset.
    Args:
    n_samples : number of generated samples.
    spectral: simulated sensor spectral.
    save_dir: samples save dir.
    """

    n_observations = len(spectral)
    ys = np.zeros((n_samples, n_observations + 1),dtype = np.float32)
    config_list = [['g', 3, 2, 0, 7], ['g', 35, 30, 5, 75], ['u', 1.3, 2.5], ['g', 0.02, 0.01, 0.002, 0.05], 
                   ['g', 0.005, 0.001, 0.001, 0.03],['u', 40, 70], ['u', 0, 1]]
    samples = batch_samples_generator(config_list, n_samples)
    tts_samples = samples_generator(['u', 40, 70], n_samples)
    ys = np.zeros((samples.shape[0], n_observations + 1), dtype = np.float32)

    if cov_function is None :
        for i in range(n_samples):
            cur_y = retrieval_model(samples[i, :], [tts_samples[i], 0, 0, spectral])
            ys[i, :-1] = cur_y + noise_generator_ratio(cur_y, 0.12, n_observations, 0.04) + noise_generator(n_observations, 0.01, 0.03)
    else:
        for i in range(n_samples):
            cur_y = cov_function(retrieval_model(samples[i, :], [tts_samples[i], 0, 0, np.arange(400, 2501, 1)]))
            ys[i, :-1] = cur_y + noise_generator_ratio(cur_y, 0.12, n_observations, 0.04) + noise_generator(n_observations, 0.01, 0.03)
    
    ys[:, n_observations] = tts_samples

    np.save('samples\\lai_l8_n_cw', samples[:, 0])
    np.save('samples\\cab_l8_n_cw', samples[:, 1])
    np.save('samples\\ys_l8_n_cw', ys)
    print('Generated.')

def train_nn(weights_save_dir = 'weights\\landsat_8\\lai_cw', samples_dir = 'samples\\lai_l8_n_cw.npy', reflectance_dir = 'samples\\ys_l8_n_cw.npy'):

    samples = np.load(samples_dir)
    ys = np.load(reflectance_dir)
    n_observations = ys.shape[1]

    mean_weight_path = os.path.join(weights_save_dir, 'mean_weight')
    mean_sc_path = os.path.join(weights_save_dir, 'mean_sc')
    var_weight_path = os.path.join(weights_save_dir, 'var_weight')
    var_sc_path = os.path.join(weights_save_dir, 'var_sc')
    if os.path.exists(weights_save_dir) is False :
        os.mkdir(weights_save_dir)

    retriever_mean = Retriever(alpha = 2, regular = 0.001)
    # retriever_mean.load_model(mean_weight_path, mean_sc_path)
    retriever_mean.train(ys, samples, 3000, 30000, n_observations, 2, 128, 0.001)
    retriever_mean.save_model(mean_weight_path, mean_sc_path)

    pred_lai = retriever_mean.predict(ys)
    lai_vars = np.power((samples - pred_lai), 2)
    retriever_var = Retriever(alpha = 2, variance = True, regular = 0.001)
    retriever_var.train(ys, lai_vars, 3000, 50000, n_observations,  2, 128, 0.001)
    retriever_var.save_model(var_weight_path, var_sc_path)

if __name__ == "__main__":
    # landsat-8 [482, 561, 655, 865, 1609, 2201]
    # sentinel-2 [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]
    # sheet_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'] # L8
    # sheet_names = ['Blue', 'Green', 'Red', 'VR1', 'VR2', 'VR3', 'NIR','NarrowNIR', 'SWIR1', 'SWIR2'] # S2
    # cov = build_srf_covolution_func('SRF\Sentinel2SRF.xlsx', sheet_names, 'Wavelength', 'srf')    
    
    # data_generator(300000, 
    #             spectral = [482, 561, 655, 865, 1609, 2201],
    #             cov_function = None)
    train_nn()
    test_nn(300, spectral = [482, 561, 655, 865, 1609, 2201], cov_function = None)
    print(".")

