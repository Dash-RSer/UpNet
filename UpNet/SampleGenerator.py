from model import model_generator
import numpy as np
from tools import samples_generator, noise_generator, concatenate_array

class SampleGenerator(object):
    def __init__(self, model, designs, priors_parameter, noise_std, noisecutrange):
        self.designs = designs
        self.RTM_model = model
        self.priors_parameter = priors_parameter
        self.N_observations = self._get_n_observations()
        self.N_sensor = len(self.designs)
        self.N_paras = len(self.priors_parameter)
        self.noise_std = noise_std
        self.cut_range = noisecutrange

    def _get_n_observations(self):
        N_observations = 0
        for i in range(len(self.designs)):
            N_observations = N_observations + len(self.designs[i][3])
        return N_observations

    def generate(self, n_samples):
        """
        return samples as [n_samples, n_paras] and reflectance as [n_samples, n_observations]
        """
        ys = np.zeros((n_samples, self.N_observations), dtype = np.float32)
        samples_matrix = np.zeros((n_samples, self.N_paras), dtype = np.float32)
        for i in range(self.N_paras):
            samples_matrix[:, i] = samples_generator(self.priors_parameter[i], n_samples)
        for i in range(n_samples):
            real_paras = samples_matrix[i, :]
            y_real = []
            for j in range(self.N_sensor):
                y = self.RTM_model(real_paras, self.designs[j])
                y_real.append(y + noise_generator(len(self.designs[j][3]), self.noise_std, self.cut_range))
            ys[i, :] = concatenate_array(np.array(y_real), self.N_observations)

        return samples_matrix, ys