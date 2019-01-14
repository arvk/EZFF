"""Interface to Q-Chem, the ab initio quantum chemistry package"""
import numpy as np
import xtal
from ezff.utils import convert_units as convert

def read_ground_state(outfilename):
    """
    Read-in a single converged structure from a QChem optimization run

    :param outfilename: Filename for ``stdout`` from the QChem job
    :type outfilename: str
    :returns: ``xtal`` tranjectory with a single snapshot with the converged structure and ground-state energy
    """
    structure = xtal.AtTraj()
    snapshot = structure.create_snapshot(xtal.Snapshot)
    outfile = open(outfilename,'r')

    # Read energies
    for line in outfile:
        if 'Final energy is' in line:
            energy_in_Hartrees = float(line.strip().split()[-1])
            break

    snapshot.energy = energy_in_Hartrees * convert.energy['Ha']['eV']

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

    outfile.close()

    return structure



def read_scan(outfilename):
    """
    Read-in a multiple partially-converged structures from a PES scan (including bond-scans, angle-scans and dihedral-scans)

    :param outfilename: Single filename for ``stdout`` from the QChem PES scan job or a list of filenames for ``stdout`` files from partial QChem PES scan jobs
    :type outfilename: str

    :returns: ``xtal`` trajectory object with structures and converged energies along the PES scan as individual snapshots
    """
    structure = xtal.AtTraj()
    energies = []

    if isinstance(outfilename, list):
        listfilenames = outfilename
    else:
        listfilenames = [outfilename]

    for single_outfilename in list(listfilenames):
        # Read energies
        outfile = open(single_outfilename,'r')
        for line in outfile:
            if 'Final energy is' in line:
                energy_in_Hartrees = float(line.strip().split()[-1])
                energies.append(energy_in_Hartrees * convert.energy['Ha']['eV'])
        outfile.close()

        # Read structure
        outfile = open(single_outfilename,'r')
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
