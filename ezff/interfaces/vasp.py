"""Interface to VASP, the Vienna Ab initio Simulation Package"""
import numpy as np
import xtal



def read_atomic_structure(structure_file):
    """
    Read-in atomic structure. Currently only VASP POSCAR/CONTCAR files are supported

    :param structure_file: Filename of the atomic structure file
    :type structure_file: str
    :returns: xtal trajectory with the structure in the first snapshot
    """
    structure = xtal.AtTraj(verbose=False)

    if ('POSCAR' in structure_file) or ('CONTCAR' in structure_file):
        structure.read_snapshot_vasp(structure_file)

    return structure



def read_phonon_dispersion(phonon_dispersion_file):
    """
    Read-in ground-truth phonon dispersion curve from VASP+Phonopy

    :param phonon_dispersion_file: Filename for the VASP + Phonopy phonon dispersion
    :type phonon_dispersion_file: str
    :returns: 2D np.array of phonon dispersion values
    """
    f = open(phonon_dispersion_file,'r')
    commentline = f.readline()
    commentline = f.readline()
    commentline = f.readline()
    segment, band, full_dispersion = [], [], []
    prevdata = 'NOT EMPTY'
    for line in f:
        data = line.strip()
        if data == '' and prevdata == '':
            full_dispersion.append(band)
            band = []
        elif data == '':
            band.append(segment)
            segment = []
        else:
            segment.append(float(data.split()[-1]))
        prevdata = data
    f.close()

    g = np.ravel(full_dispersion[0])
    for i in range(1,len(full_dispersion)):
        fd = np.ravel(full_dispersion[i])
        if len(fd):
            g = np.vstack((g,fd))

    return g
