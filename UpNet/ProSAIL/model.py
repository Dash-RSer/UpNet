# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

import numpy as np
import prosail

def multi_band_model(n, cab, car, cw, cm, lai, ala, hotspot, psoil, geometry, sensor_cw = [[840]]):
    """
    """
    tts, tto, psi = geometry
    sensor_cw = np.array(sensor_cw)
    res = prosail.run_prosail(
        n = n, cab = cab,car = car, cbrown = 0,cw = cw, cm = cm,lai = lai,
        lidfa = ala, hspot = hotspot,tts = tts, tto = tto,psi = psi,psoil = psoil, rsoil=0.8)
    selected_reflectance = res[sensor_cw - 400]
    return selected_reflectance


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

def model_generator_omega(settings):
    """
    settings: such as ['free', 40, 'free', ...] as the order
    [lai , cab, car, n, cw, cm, ala, hotspot, psoil]
    """
    N = 10
    N_free = settings.count('free')
    N_fixed = N - N_free
    for i in range(N):
        if settings[i] == 'free':
            settings[i] = 9999
    
    settings = np.array(settings, dtype = np.float32)
    index = np.where(settings == 9999)
    
    def model(parameters, designs):
        settings[index] = parameters
        lai, cab, car, n, cw, cm, ala, hotspot, psoil, omega = settings
        geometry = designs[:3]
        sensor_cw = designs[3]
        ref = multi_band_model(n, cab, car, cw, cm, lai*omega, ala, hotspot, psoil, geometry, sensor_cw)
        return ref
    return model

if __name__ == "__main__":
    model = model_generator(['free', 30, 8, 2, 0.01, 'free', 60, 0.3, 0.2, 'free'])
    bands = [800]
    # angles = np.arange(-70, 80, 1)
    # ds = []
    # for angle in angles:
    #     if angle < 0:
    #         psi = 0
    #         angle = -angle
    #     else:
    #         psi = 180
    #     d = np.abs(model([2, 0.01, 0.01, 60+1], [20, angle, psi,[550]]) - model([2, 0.01, 0.01, 60-1], [20, angle, psi,[550]]))/2
    #     print(angle, psi)
    #     ds.append(d)
    
    print(".")

    
# def multi_band_model(n, cab, car, cw, cm, lai, ala, hotspot, psoil, geometry, sensor_cw = [[840, 1]]):
#     """
#     """
#     tts, tto, psi = geometry
#     sensor_cw = np.array(sensor_cw)
#     res = prosail.run_prosail(
#         n = n, cab = cab,car = car, cbrown = 0,cw = cw, cm = cm,lai = lai,
#         lidfa = ala, hspot = hotspot,tts = tts, tto = tto,psi = psi,psoil = psoil, rsoil=0.8)
#     reflectance
#     for i in range(len(sensor_cw)):
#     selected_reflectance = res[sensor_cw - 400]
#     return selected_reflectance

#def multi_band_model_bandwidth(n, cab, car, cw, cm, lai, ala, hotspot, psoil, geometry, sensor_cw = [[840, 1]]):

#     tts, tto, psi = geometry
#     sensor_cw = np.array(sensor_cw)
#     res = prosail.run_prosail(
#         n = n, cab = cab,car = car, cbrown = 0,cw = cw, cm = cm,lai = lai,
#         lidfa = ala, hspot = hotspot,tts = tts, tto = tto,psi = psi,psoil = psoil, rsoil=0.8)
#     reflectance = np.array(len(sensor_cw), dtype = np.float32)
#     for i in range(len(sensor_cw)):
#         cw = sensor_cw[i][0]
#         bw = sensor_cw[i][1]
#         reflectance[i] = np.mean(res[cw - bw - 400: cw + bw - 400])
#     return reflectance