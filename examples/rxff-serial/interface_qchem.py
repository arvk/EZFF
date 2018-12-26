"""Interface to Q-Chem, the ab initio quantum chemistry package"""
import numpy as np
import xtal

energy_conversion = {'HatoeV':27.211, 'Hatokcal': 627.5}

def read_ground_state(outfilename):
    structure = xtal.AtTraj()
    snapshot = structure.create_snapshot(xtal.Snapshot)
    outfile = open(outfilename,'r')

    # Read energies
    for line in outfile:
        if 'Final energy is' in line:
            energy_in_Hartrees = float(line.strip().split()[-1])
            break

    snapshot.energy = energy_in_Hartrees*energy_conversion['HatoeV']

    for line in outfile:
        if 'OPTIMIZATION CONVERGED' in line:
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            # Atomic coordinates start here
            while True:
                coords = outfile.readline()
                if coords.strip()=='':
                    break
                atom = snapshot.create_atom(xtal.Atom)
                atom.element, atom.cart = coords.strip().split()[1], np.array(list(map(float,coords.strip().split()[2:5])))
            break

    for line in outfile:
        if 'Ground-State Mulliken Net Atomic Charges' in line:
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            for atom in snapshot.atomlist:
                atom.charge = outfile.readline().strip().split()[2]
            break

    return structure



def read_scan(outfilename):
    structure = xtal.AtTraj()

    # Read energies
    energies = []
    outfile = open(outfilename,'r')
    for line in outfile:
        if 'Final energy is' in line:
            energy_in_Hartrees = float(line.strip().split()[-1])
            energies.append(energy_in_Hartrees*energy_conversion['HatoeV'])
    outfile.close()

    # Read structure
    outfile = open(outfilename,'r')
    for line in outfile:
        if 'OPTIMIZATION CONVERGED' in line:
            snapshot = structure.create_snapshot(xtal.Snapshot)
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            # Atomic coordinates start here
            while True:
                coords = outfile.readline()
                if coords.strip()=='':
                    break
                atom = snapshot.create_atom(xtal.Atom)
                atom.element, atom.cart = coords.strip().split()[1], np.array(list(map(float,coords.strip().split()[2:5])))

    outfile.close()

    for counter, snapshot in enumerate(structure.snaplist):
        snapshot.energy = energies[counter]

    return structure
