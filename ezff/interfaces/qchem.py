"""Interface to Q-Chem, the ab initio quantum chemistry package"""
import numpy as np
import xtal
from ezff.utils import convert_units as convert



def read_structure(outfilename):
    """
    Read-in a multiple partially-converged structures from a PES scan (including bond-scans, angle-scans and dihedral-scans)

    :param outfilename: Single filename for ``stdout`` from the QChem PES scan job or a list of filenames for ``stdout`` files from partial QChem PES scan jobs
    :type outfilename: str

    :returns: ``xtal`` trajectory object with structures and converged energies along the PES scan as individual snapshots
    """
    structure = xtal.AtTraj()
    structure.box = np.zeros((3,3))
    energies = []

    if isinstance(outfilename, list):
        listfilenames = outfilename
    else:
        listfilenames = [outfilename]

    for single_outfilename in list(listfilenames):
        # Read structure
        outfile = open(single_outfilename,'r')
        while True:
            line = outfile.readline()
            if not line: # break at EOF
                break
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
                xpos_this_snapshot = [atom.cart[0] for atom in snapshot.atomlist]
                ypos_this_snapshot = [atom.cart[1] for atom in snapshot.atomlist]
                zpos_this_snapshot = [atom.cart[2] for atom in snapshot.atomlist]
                offset_vector = 0.0 - np.array([np.amin(xpos_this_snapshot), np.amin(ypos_this_snapshot), np.amin(zpos_this_snapshot)])
                snapshot.move(offset_vector)
        outfile.close()

    # Since the QChem structure is often non-periodic, we set up a box that is 50 Angstrom larger than the maximum extent of atoms
    xpos = [atom.cart[0] for atom in snapshot.atomlist for snapshot in structure.snaplist]
    ypos = [atom.cart[1] for atom in snapshot.atomlist for snapshot in structure.snaplist]
    zpos = [atom.cart[2] for atom in snapshot.atomlist for snapshot in structure.snaplist]

    structure.box[0][0] = np.amax(xpos) - np.amin(xpos) + 50.0
    structure.box[1][1] = np.amax(ypos) - np.amin(ypos) + 50.0
    structure.box[2][2] = np.amax(zpos) - np.amin(zpos) + 50.0

    return structure



def read_energy(outfilename):
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
    return np.array(energies)



def read_atomic_charges(outfilename):
    """
    Read-in a multiple partially-converged structures from a PES scan (including bond-scans, angle-scans and dihedral-scans)

    :param outfilename: Single filename for ``stdout`` from the QChem PES scan job or a list of filenames for ``stdout`` files from partial QChem PES scan jobs
    :type outfilename: str

    :returns: ``xtal`` trajectory object with structures and converged energies along the PES scan as individual snapshots
    """
    structure = xtal.AtTraj()
    structure.box = np.zeros((3,3))
    energies = []

    if isinstance(outfilename, list):
        listfilenames = outfilename
    else:
        listfilenames = [outfilename]

    for single_outfilename in list(listfilenames):
        # Read structure
        outfile = open(single_outfilename,'r')
        while True:
            line = outfile.readline()
            if not line: # break at EOF
                break

            if 'Ground-State Mulliken Net Atomic Charges' in line:
                charge_array = []
                dummyline = outfile.readline()
                dummyline = outfile.readline()
                dummyline = outfile.readline()
                while True:
                    charges = outfile.readline().strip().split()
                    if charges[0][0] == '-':
                        break
                    charge_array.append(float(charges[2]))

            if 'OPTIMIZATION CONVERGED' in line:
                snapshot = structure.create_snapshot(xtal.Snapshot)
                dummyline = outfile.readline()
                dummyline = outfile.readline()
                dummyline = outfile.readline()
                dummyline = outfile.readline()
                # Atomic coordinates start here
                atomID = 0
                while True:
                    coords = outfile.readline()
                    if coords.strip()=='':
                        break
                    atom = snapshot.create_atom(xtal.Atom)
                    atom.element, atom.cart = coords.strip().split()[1], np.array(list(map(float,coords.strip().split()[2:5])))
                    atom.charge = charge_array[atomID]
                    atomID += 1

        outfile.close()
    return structure
