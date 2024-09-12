import numpy as np
import matplotlib.pyplot as plt
from tools import samples_generator, noise_generator_ratio
from SRFCovolution import build_srf_covolution_func
from model import model_generator
from NN_model import Retriever
from MCMC import MCMC_inference
import warnings
import os
warnings.filterwarnings('ignore')
import time
import pandas as pd
from sklearn.metrics import r2_score

df = pd.read_excel('total.xlsx', sheet_name = 'filted')
reflectance_matrix = np.concatenate([df.to_numpy(dtype = np.float32)[:, 1:7], \
                                     df['tts'].to_numpy(dtype = np.float32)[:].reshape(-1, 1)], axis = 1)
lai_effs = df['LAIeff'].to_numpy(dtype = np.float32)[:]
N_samples = lai_effs.shape[0]

weights_dir = 'weights\\real\l8_n'
mean_weight_path = os.path.join(weights_dir, 'mean_weight')
mean_sc_path = os.path.join(weights_dir, 'mean_sc')
var_weight_path = os.path.join(weights_dir, 'var_weight')
var_sc_path = os.path.join(weights_dir, 'var_sc')

# Neural Network
retriever_mean = Retriever()
retriever_mean.load_model(mean_weight_path, mean_sc_path)
y_pred_mean = retriever_mean.predict(reflectance_matrix)
retriever_var = Retriever(variance = True)
retriever_var.load_model(var_weight_path, var_sc_path)
y_pred_var = retriever_var.predict(reflectance_matrix)

plt.scatter(y_pred_mean, lai_effs, s = 3.5)
plt.xlim(0,6)
plt.ylim(0,6)
plt.plot([0,6], [0,6], color = 'r')
plt.show()
# mcmc
sheet_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
cov = build_srf_covolution_func('SRF\Ball_BA_RSR.v1.2.xlsx', sheet_names, 'Wavelength', 'srf') 
mcmc_pred_lais = []
mcmc_pred_stds = []
start = time.time()
for i in range(N_samples):
    mean, std = MCMC_inference(reflectance_matrix[i, :], parameter_index = 0, cov_func = cov)
    mcmc_pred_lais.append(mean)
    mcmc_pred_stds.append(std)
end = time.time()
print('MCMC time:', end - start)
# 11921.765077590942 s
print(".")

