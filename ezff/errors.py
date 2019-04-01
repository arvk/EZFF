"""This module provide general functions for EZFF"""
import os
import sys
import xtal
import numpy as np
from platypus import Problem, unique, nondominated, NSGAII, NSGAIII, IBEA, PoolEvaluator
from platypus.types import Real, Integer
from platypus.operators import InjectedPopulation, GAOperator, SBX, PM

def error_phonon_dispersion(md_disp, gt_disp, weights='uniform', verbose=False):
    """
    Calculate error between MD-computed phonon dispersion and the ground-truth phonon dispersion with user-defined weighting schemes

    :param md_disp: MD-computed phonon dispersion curve
    :type md_disp: 2D np.array

    :param gt_disp: Ground-truth phonon dispersion curve
    :type gt_disp: 2D np.array

    :param weights: User-defined weighting scheme for calculating errors provided as a list of numbers, one per band. Possible values are
                    ``uniform`` - where errors from all bands are equally weighted,
                    ``acoustic`` - where errors from lower-frequency bands are assigned greater weights, and
                    `list` - 1-D list of length equal to number of bands
    :type weights: str `or` list

    :param verbose: Deprecated option for verbosity of error calculation routine
    :type verbose: bool
    """
    # Perform sanity check. Number of bands should be equal between the two structures
    if not len(md_disp) == len(gt_disp):
        raise ValueError("MD and ground truth dispersions have different number of bands")
        return

    # Create array of weights - one value per band
    num_band = len(md_disp)
    if weights == 'uniform':
        W = np.ones(num_band)
    elif weights == 'acoustic':
        maxfreq = np.amax(gt_disp)
        W = np.reciprocal((np.mean(gt_disp, axis=1)/maxfreq) + 0.1)
    elif isinstance(weights,list) or isinstance(weights,np.ndarray):
        if len(weights) == num_band:
            W = np.array(weights)
        else:
            raise ValueError("Number of provided weight values is different from number of bands! Aborting")

    # Compute the RMS error between dispersions
    rms_error = 0.0
    num_k_gt = len(gt_disp[0])
    scaling = num_k_gt/100.0
    for band_index in range(0, len(gt_disp)):
        interp_md_band = np.interp(np.arange(0, num_k_gt), np.arange(0, 100)*scaling, md_disp[band_index])
        rms_error += np.linalg.norm(interp_md_band - gt_disp[band_index]) * W[band_index]

    rms_error /= (num_k_gt * num_band)
    return rms_error



def error_PES_scan(md_scan, gt_scan, weights='uniform', verbose=False):
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
    if not len(md_scan) == len(gt_scan):
        raise ValueError("MD and ground truth PES have different number of points! Aborting")
        return

    md_scan = np.array(md_scan)
    gt_scan = np.array(gt_scan)

    num_pes = len(gt_scan)
    W = np.ones(num_pes)
    if weights == 'uniform':
        pass
    elif weights == 'minima':
        min_E = np.amin(gt_scan)
        max_E = np.amax(gt_scan)
        W = np.reciprocal(((gt_scan-min_E)/max_E) + 0.1)
    elif weights == 'dissociation':
        min_E = np.amin(gt_scan)
        max_E = np.amax(gt_scan)
        W = (9.0*(gt_scan-min_E)/max_E) + 1.0
    elif isinstance(weights,list) or isinstance(weights,np.ndarray):
        if len(weights) == len(gt_scan):
            W = np.array(weights)
        else:
            raise ValueError("Weights array and PES have different number of points! Aborting")

    # Compute the RMS error between PES
    rms_error = np.linalg.norm((md_scan - gt_scan) * W)
    return rms_error



def error_structure_distortion(outfilename, relax_atoms=False, relax_cell=False):
    if not relax_atoms:  # If atoms are not relaxed (i.e. single point calculation, then return 0.0)
        return 0.0       # In the future, add a info/warning message in the log

    if relax_atoms:      # If atoms are leaxed, then create 2 atomic trajectories, one for each of the initial and relaxed structures
        initial = xtal.AtTraj()
        initial.abc = np.array([0.0,0.0,0.0])
        initial.ang = np.array([0.0,0.0,0.0])
        initial_snapshot = initial.create_snapshot(xtal.Snapshot)
        relaxed = xtal.AtTraj()
        relaxed.abc = np.array([0.0,0.0,0.0])
        relaxed.ang = np.array([0.0,0.0,0.0])
        relaxed_snapshot = relaxed.create_snapshot(xtal.Snapshot)

        # Read number of atoms
        outfile = open(outfilename, 'r')
        for line in outfile:
            if 'Number of irreducible atoms/shells' in line.strip():
                num_atoms = int(line.strip().split()[-1])
        outfile.close()

        for atomID in range(num_atoms):
            initial_snapshot.create_atom(xtal.Atom)
            initial_snapshot.atomlist[atomID].cart = np.array([0.0,0.0,0.0])
            initial_snapshot.atomlist[atomID].fract = np.array([0.0,0.0,0.0])
            relaxed_snapshot.create_atom(xtal.Atom)
            relaxed_snapshot.atomlist[atomID].cart = np.array([0.0,0.0,0.0])
            relaxed_snapshot.atomlist[atomID].fract = np.array([0.0,0.0,0.0])

        if relax_cell:     # In atoms are relaxed, and simulation cell is also relaxed
            convert_to_cart = True # We have to read in 2 box sizes, one for the initial cell and one for the relaxed cell
            outfile = open(outfilename, 'r')
            for oneline in outfile:
                if 'Comparison of initial and final' in oneline:
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    while True:
                        data = outfile.readline().strip().split()
                        if data[0].isdigit():
                            atomID = int(data[0])-1
                            if data[1] == 'x':
                                axisID = 0
                            elif data[1] == 'y':
                                axisID = 1
                            else:
                                axisID = 2

                            if data[5] == 'Cartesian':
                                initial_snapshot.atomlist[atomID].cart[axisID] = float(data[2])
                                relaxed_snapshot.atomlist[atomID].cart[axisID] = float(data[3])
                                convert_to_cart = False
                            elif data[5] == 'Fractional':
                                initial_snapshot.atomlist[atomID].fract[axisID] = float(data[2])
                                relaxed_snapshot.atomlist[atomID].fract[axisID] = float(data[3])
                        elif data[0][0] == '-':
                            break
                    break
            outfile.close()

            if convert_to_cart: # READ TWO CELL SIZES - ONE FOR THE INITIAL CELL, ONE FOR THE RELAXED CELL
                outfile = open(outfilename, 'r')
                for oneline in outfile:
                    if 'Comparison of initial and final' in oneline:
                        dummyline = outfile.readline()
                        dummyline = outfile.readline()
                        dummyline = outfile.readline()
                        dummyline = outfile.readline()
                        while True:
                            data = outfile.readline().strip().split()
                            if data[0] == 'a':
                                initial.abc[0], relaxed.abc[0] = float(data[1]), float(data[2])
                            elif data[0] == 'b':
                                initial.abc[1], relaxed.abc[1] = float(data[1]), float(data[2])
                            elif data[0] == 'c':
                                initial.abc[2], relaxed.abc[2] = float(data[1]), float(data[2])
                            elif data[0] == 'alpha':
                                initial.ang[0], relaxed.ang[0] = float(data[1]), float(data[2])
                            elif data[0] == 'beta':
                                initial.ang[1], relaxed.ang[1] = float(data[1]), float(data[2])
                            elif data[0] == 'gamma':
                                initial.ang[2], relaxed.ang[2] = float(data[1]), float(data[2])
                            elif data[0][0] == '-':
                                break
                        break
                outfile.close()
                initial.abc_to_box()
                relaxed.abc_to_box()
                initial.make_dircar_matrices()
                relaxed.make_dircar_matrices()
                initial.dirtocar()
                relaxed.dirtocar()
                relaxed.move(initial.snaplist[0].atomlist[0].cart  - relaxed.snaplist[0].atomlist[0].cart)
                error = 0.0
                for i in range(len(initial.snaplist[0].atomlist)):
                    dr = initial.snaplist[0].atomlist[i].cart - relaxed.snaplist[0].atomlist[i].cart
                    error += np.inner(dr, dr)
            else:
                relaxed.move(initial.snaplist[0].atomlist[0].cart  - relaxed.snaplist[0].atomlist[0].cart)
                error = 0.0
                for i in range(len(initial.snaplist[0].atomlist)):
                    dr = initial.snaplist[0].atomlist[i].cart - relaxed.snaplist[0].atomlist[i].cart
                    error += np.inner(dr, dr)
            return error

        else:     # IF THE CELL IS NOT RELAXED. JUST THE ATOMS ARE RELAXED
            convert_to_cart = True
            outfile = open(outfilename, 'r')
            for oneline in outfile:
                if 'Comparison of initial and final' in oneline:
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    while True:
                        data = outfile.readline().strip().split()
                        if data[0].isdigit():
                            atomID = int(data[0])-1
                            if data[1] == 'x':
                                axisID = 0
                            elif data[1] == 'y':
                                axisID = 1
                            else:
                                axisID = 2

                            if data[5] == 'Cartesian':
                                initial_snapshot.atomlist[atomID].cart[axisID] = float(data[2])
                                relaxed_snapshot.atomlist[atomID].cart[axisID] = float(data[3])
                                convert_to_cart = False
                            elif data[5] == 'Fractional':
                                initial_snapshot.atomlist[atomID].fract[axisID] = float(data[2])
                                relaxed_snapshot.atomlist[atomID].fract[axisID] = float(data[3])
                        elif data[0][0] == '-':
                            break
                    break
            outfile.close()

            if convert_to_cart:
                outfile = open(outfilename, 'r')   ### READ A SINGLE BOX SIZE FOR THE INITIAL AND RELAXED STRUCTURES
                for oneline in outfile:
                    if 'Cartesian lattice vectors (Angstroms)' in oneline:
                        dummyline = outfile.readline()
                        for i in range(3):
                            initial.box[i][0:3] = list(map(float,outfile.readline().strip().split()))
                outfile.close()
                relaxed.box = initial.box
                initial.make_dircar_matrices()
                relaxed.make_dircar_matrices()
                initial.dirtocar()
                relaxed.dirtocar()
                relaxed.move(initial.snaplist[0].atomlist[0].cart  - relaxed.snaplist[0].atomlist[0].cart)
                error = 0.0
                for i in range(len(initial.snaplist[0].atomlist)):
                    dr = initial.snaplist[0].atomlist[i].cart - relaxed.snaplist[0].atomlist[i].cart
                    error += np.inner(dr, dr)
            else:
                relaxed.move(initial.snaplist[0].atomlist[0].cart  - relaxed.snaplist[0].atomlist[0].cart)
                error = 0.0
                for i in range(len(initial.snaplist[0].atomlist)):
                    dr = initial.snaplist[0].atomlist[i].cart - relaxed.snaplist[0].atomlist[i].cart
                    error += np.inner(dr, dr)
            return error
