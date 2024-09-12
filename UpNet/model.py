# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

import numpy as np
from ProSAIL.RTMS import ProSAIL

def multi_band_model(n, cab, car, cw, cm, lai, ala, hotspot, psoil, geometry, sensor_cw = [840]):
    """
    """
    tts, tto, psi = geometry
    sensor_cw = np.array(sensor_cw)
    res = ProSAIL(n, cab, car, cw, cm, lai, ala, hotspot, 0.8, psoil, tts, tto, psi, sensor_cw)
    return res

def model_generator(settings):
    """
    settings: such as ['free', 40, 'free', ...] as the order
    [lai , cab, car, n, cw, cm, ala, hotspot, psoil]
    """
    N = 9
    N_free = settings.count('free')
    N_fixed = N - N_free
    for i in range(N):
        if settings[i] == 'free':
            settings[i] = 9999
    
    settings = np.array(settings, dtype = np.float32)
    index = np.where(settings == 9999)
    def model(parameters, designs):
        settings[index] = parameters
        lai, cab, car, n, cw, cm, ala, hotspot, psoil = settings
        geometry = designs[:3]
        sensor_cw = designs[3]
        ref = multi_band_model(n, cab, car, cw, cm, lai, ala, hotspot, psoil, geometry, sensor_cw)
        return ref
    return model

if __name__ == "__main__":
    print(".")
