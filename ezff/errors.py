"""This module provide functions for computing errors from previously completed MD runs"""
import os
import sys
import xtal
import numpy as np
# from platypus import Problem, unique, nondominated, NSGAII, NSGAIII, IBEA, PoolEvaluator
# from platypus.types import Real, Integer
# from platypus.operators import InjectedPopulation, GAOperator, SBX, PM



def error_phonon_dispersion(MD=None, GT=None, weights='uniform', verbose=False):
    """
    Calculate error between MD-computed phonon dispersion and the ground-truth phonon dispersion with user-defined weighting schemes

    :param MD: MD-computed phonon dispersion curve
    :type MD: 2D np.array

    :param GT: Ground-truth phonon dispersion curve
    :type GT: 2D np.array

    :param weights: User-defined weighting scheme for calculating errors provided as a list of numbers, one per band. Possible values are
                    ``uniform`` - where errors from all bands are equally weighted,
                    ``acoustic`` - where errors from lower-frequency bands are assigned greater weights, and
                    `list` - 1-D list of length equal to number of bands
    :type weights: str `or` list

    :param verbose: Deprecated option for verbosity of error calculation routine
    :type verbose: bool
    """
    # Perform sanity check. Number of bands should be equal between the two structures
    if not len(MD) == len(GT):
        raise ValueError("MD and ground truth dispersions have different number of bands")
        return np.nan

    # Create array of weights - one value per band
    num_band = len(MD)
    if weights == 'uniform':
        W = np.ones(num_band)
    elif weights == 'acoustic':
        maxfreq = np.amax(GT)
        W = np.reciprocal((np.mean(GT, axis=1)/maxfreq) + 0.1)
    elif isinstance(weights,list) or isinstance(weights,np.ndarray):
        if len(weights) == num_band:
            W = np.array(weights)
        else:
            raise ValueError("Number of provided weight values is different from number of bands! Aborting")

    # Compute the RMS error between dispersions
    try:
        rms_error = 0.0
        num_k_gt = len(GT[0])
        scaling = num_k_gt/100.0
        for band_index in range(0, len(GT)):
            interp_md_band = np.interp(np.arange(0, num_k_gt), np.arange(0, 100)*scaling, MD[band_index])
            rms_error += np.linalg.norm(interp_md_band - GT[band_index]) * W[band_index]

        rms_error /= (num_k_gt * num_band)
    except:
        rms_error = np.nan

    return rms_error



def error_structure_distortion(MD=None, GT=None):
    """
    Calculate error due to relaxation of atoms in the initial structure. The error is the sum of root mean square displacement of atoms.

    :param MD: Relaxed structure after MD run
    :type MD: xtal.AtTraj object

    :param GT: Initial Ground-Truth structure used as input for MD calculations
    :type GT: xtal.AtTraj object
    """
    # Sanity checks -- Both inputs should be AtTraj objects
    if not isinstance(MD, xtal.AtTraj) and isinstance(GT, xtal.AtTraj):
        print('ERROR_STRUCTURE_DISTORTION: Please provide xtal.AtTraj objects for comparison')
        return np.nan

    if not (len(GT.snaplist) == len(MD.snaplist)):
        print('Different number of structures in MD and Ground-Truth data')
        return np.nan

    if not np.allclose(MD.box, GT.box):
        print('ERROR: Cell sizes and shapes are different for MD and ground-truth. Please use error_lattice_constant instead')
        return np.nan

    try:
        error = 0.0
        for snapID in range(len(GT.snaplist)):
            errors_this_snapshot = []
            for atomID in range(len(GT.snaplist[snapID].atomlist)):
                error_this_atom = np.linalg.norm(GT.snaplist[snapID].atomlist[atomID].cart - MD.snaplist[snapID].atomlist[atomID].cart)
                errors_this_snapshot.append(error_this_atom)
            error += np.linalg.norm(np.array(errors_this_snapshot))
    except:
        error = np.nan

    return error



def error_lattice_constant(MD=None, GT=None):
    """
    Calculate error due to optimization of lattice constants in the initial structure.

    :param MD: Relaxed structure after MD run
    :type MD: xtal.AtTraj object

    :param GT: Initial Ground-Truth structure used as input for MD calculations
    :type GT: xtal.AtTraj object
    """
    # Sanity checks -- Both inputs should be AtTraj objects
    if not isinstance(MD, xtal.AtTraj) and isinstance(GT, xtal.AtTraj):
        print('ERROR_LATTICE_CONSTANT: Please provide xtal.AtTraj objects for comparison')
        return

    try:
        MD.make_dircar_matrices()
        GT.make_dircar_matrices()
        MD.box_to_abc()
        GT.box_to_abc()
        abc = MD.abc - GT.abc
        ang = MD.ang - GT.ang
    except:
        abc = np.array([np.nan, np.nan, np.nan])
        ang = np.array([np.nan, np.nan, np.nan])

    return abc, ang



def error_atomic_charges(MD=None, GT=None):
    """
    Calculate error due to difference between MD-computed atomic charges and ground-truth atomic charges

    :param MD: Relaxed structure after MD run
    :type MD: xtal.AtTraj object

    :param GT: Initial Ground-Truth structure used as input for MD calculations
    :type GT: xtal.AtTraj object
    """
    # Sanity checks -- Both inputs should be AtTraj objects
    if not isinstance(MD, xtal.AtTraj) and isinstance(GT, xtal.AtTraj):
        print('ERROR_ATOMIC_CHARGES: Please provide xtal.AtTraj objects for comparison')
        return np.nan

    if not (len(GT.snaplist) == len(MD.snaplist)):
        print('Different number of structures in MD and Ground-Truth data')
        return np.nan

    error_array = []
    for snapID in range(len(GT.snaplist)):
        GT_charges = np.array([atom.charge for atom in GT.snaplist[snapID].atomlist])
        MD_charges = np.array([atom.charge for atom in MD.snaplist[snapID].atomlist])
        try:
            error_this_snapshot = np.linalg.norm(GT_charges - MD_charges)
        except:
            return np.nan
        error_array.append(error_this_snapshot)

    return np.sum(error_array)



def error_energy(MD, GT, weights='uniform', verbose=False):
    """
    Calculate error between MD-computed potential energy surface and the ground-truth potential energy surface with user-defined weighting schemes

    :param md_disp: MD-computed potential energy surface
    :type md_disp: 1D np.array

    :param gt_disp: Ground-truth potential energy surface
    :type gt_disp: 1D np.array

    :param weights: User-defined weighting scheme for calculating errors. Possible values are
                    ``uniform`` - where errors from all points on the PES are weighted equally,
                    ``minima`` - where errors from lower-energy points are assigned greater weights,
                    ``dissociation`` - where errors from highest-energy points are assigned greater weights, and
                    `list` - 1-D list of length equal to number of points on the PES scans
    :type weights: str `or` list

    :param verbose: Deprecated option for verbosity of error calculation routine
    :type verbose: bool
    """
    # Perform sanity check. Number of bands should be equal between the two structures
    if not len(MD) == len(GT):
        raise ValueError("MD and ground truth PES have different number of points! Aborting")
        return

    MD = np.array(MD)
    GT = np.array(GT)

    num_pes = len(GT)
    W = np.ones(num_pes)
    if weights == 'uniform':
        pass
    elif weights == 'minima':
        min_E = np.amin(GT)
        max_E = np.amax(GT)
        W = np.reciprocal(((GT-min_E)/max_E) + 0.1)
    elif weights == 'dissociation':
        min_E = np.amin(GT)
        max_E = np.amax(GT)
        W = (9.0*(GT-min_E)/max_E) + 1.0
    elif isinstance(weights,list) or isinstance(weights,np.ndarray):
        if len(weights) == len(GT):
            W = np.array(weights)
        else:
            raise ValueError("Weights array and PES have different number of points! Aborting")

    # Compute the RMS error between PES
    try:
        rms_error = np.linalg.norm((MD - GT) * W)
    except:
        rms_error = np.nan

    return rms_error
