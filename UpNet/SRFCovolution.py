import numpy as np
import pandas as pd
import numba as nb
import warnings
warnings.filterwarnings('ignore')

def single_band_covolution(sim_ref_cut, srf_band):
    """
    Covolute the simulated reflectance to sensor reflectance.
    """
    donominator = np.sum(srf_band)
    numerator = 0
    for i in range(len(sim_ref_cut)):
        numerator += sim_ref_cut[i] * srf_band[i]
    return numerator/donominator

def build_srf_covolution_func(excel_file, sheet_names, band_col_name, srf_col_name):
    N_bands = len(sheet_names)
    wavelengths = []
    srfs = []
    for i in range(N_bands): 
        df = pd.read_excel(excel_file, sheet_name = sheet_names[i])
        wavelengths.append(df[band_col_name].to_numpy() - 400)
        srfs.append(np.clip(df[srf_col_name].to_numpy(), 0, 1))
    
    def cov(sim_ref):
        refs = np.zeros(N_bands, dtype = np.float32)
        for i in range(N_bands):
            refs[i] = single_band_covolution(sim_ref[wavelengths[i]], srfs[i])
        return refs
    
    return cov

if __name__ == "__main__":
    from ProSAIL.RTMS import ProSAIL
    ref = ProSAIL(2,40, 8, 0.01, 0.01, 3, 60, 0.3, 0.8, 0.2, 30, 0, 0, np.arange(400, 2501, 1))
    sheet_names = ['Blue', 'Green', 'Red', 'VR1', 'VR2', 'VR3', 'NIR','NarrowNIR', 'SWIR1', 'SWIR2'] # S2
    cov = build_srf_covolution_func('SRF\Sentinel2SRF.xlsx', sheet_names, 'Wavelength', 'srf') 
    ref_s2 = cov(ref)
    ref_s2_cw = ProSAIL(2,40, 8, 0.01, 0.01, 3, 60, 0.3, 0.8, 0.2, 30, 0, 0, [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190])
    print(".")
