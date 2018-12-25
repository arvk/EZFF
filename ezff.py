"""This module provide general functions for EZFF"""
import numpy as np

def error_phonon_dispersion(md_disp, gt_disp, weights='uniform', verbose=False):
    """Calculate error between MD-computed dispersion and the ground-truth"""
    # Perform sanity check. Number of bands should be equal between the two structures
    if not len(md_disp) == len(gt_disp):
        raise ValueError("MD and ground truth dispersions have different number of bands")
        return

    # Create array of weights - one value per band
    num_band = len(md_disp)
    W = np.ones(num_band)
    if weights == 'uniform':
        pass
    elif weights == 'acoustic':
        maxfreq = np.amax(gt_disp)
        W = np.reciprocal((np.mean(gt_disp,axis=1)/maxfreq) + 0.1)

    # Compute the RMS error between dispersions
    rms_error = 0.0
    num_k_gt = len(gt_disp[0])
    scaling = num_k_gt/100.0
    for band_index in range(0,len(gt_disp)):
        interp_md_band = np.interp(np.arange(0,num_k_gt),np.arange(0,100)*scaling,md_disp[band_index])
        rms_error += np.linalg.norm(interp_md_band - gt_disp[band_index]) * W[band_index]

    rms_error /= (num_k_gt * num_band)
    return rms_error
