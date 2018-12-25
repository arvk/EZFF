"""Interface to VASP, the Vienna Ab initio Simulation Package"""
import numpy as np

def read_phonon_dispersion(phonon_dispersion_file):
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
