"""Interface to Q-Chem, the ab initio quantum chemistry package"""
import numpy as np
import xtal
import cclib
import periodictable as pt

def read_structure(outfilename):
    """
    Read-in a multiple partially-converged structures from a PES scan (including bond-scans, angle-scans and dihedral-scans)

    :param outfilename: Single filename for ``stdout`` from the QChem PES scan job or a list of filenames for ``stdout`` files from partial Gaussian PES scan jobs
    :type outfilename: str

    :returns: ``xtal`` trajectory object with structures and converged energies along the PES scan as individual snapshots
    """
    structure = xtal.AtTraj()
    structure.box = np.zeros((3,3))

    if isinstance(outfilename, list):
        listfilenames = outfilename
    else:
        listfilenames = [outfilename]

    for single_outfilename in list(listfilenames):
        # Read structure
        data = cclib.io.ccread(single_outfilename)
        atomic_numbers = data.atomnos
        for configuration in data.atomcoords:
            snapshot = structure.create_snapshot(xtal.Snapshot)
            for atom_id, atom_coord in enumerate(configuration):
                atom = snapshot.create_atom(xtal.Atom)
                atom.element = str(pt.elements[atomic_numbers[atom_id]])
                atom.cart = np.array(atom_coord)

    # Since the Gaussian structure is often non-periodic, we set up a box that is 50 Angstrom larger than the maximum extent of atoms
    xpos = [atom.cart[0] for atom in snapshot.atomlist for snapshot in structure.snaplist]
    ypos = [atom.cart[1] for atom in snapshot.atomlist for snapshot in structure.snaplist]
    zpos = [atom.cart[2] for atom in snapshot.atomlist for snapshot in structure.snaplist]

    structure.box[0][0] = np.amax(xpos) - np.amin(xpos) + 100.0
    structure.box[1][1] = np.amax(ypos) - np.amin(ypos) + 100.0
    structure.box[2][2] = np.amax(zpos) - np.amin(zpos) + 100.0

    structure.move([0.0 - np.amin(xpos), 0.0 - np.amin(ypos), 0.0 - np.amin(zpos)])
    structure.move([50.0, 50.0, 50.0])

    structure.make_dircar_matrices()
    structure.cartodir()

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
        # Read structure
        data = cclib.io.ccread(single_outfilename)
        energies.extend(data.scfenergies)

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
    charges = []

    if isinstance(outfilename, list):
        listfilenames = outfilename
    else:
        listfilenames = [outfilename]

    for single_outfilename in list(listfilenames):
        # Read structure
        data = cclib.io.ccread(single_outfilename)
        atomic_numbers = data.atomnos
        for configuration in data.atomcoords:
            snapshot = structure.create_snapshot(xtal.Snapshot)
            for atom_id, atom_coord in enumerate(configuration):
                atom = snapshot.create_atom(xtal.Atom)
                atom.element = str(pt.elements[atomic_numbers[atom_id]])
                atom.cart = np.array(atom_coord)

    # Since the Gaussian structure is often non-periodic, we set up a box that is 50 Angstrom larger than the maximum extent of atoms
    xpos = [atom.cart[0] for atom in snapshot.atomlist for snapshot in structure.snaplist]
    ypos = [atom.cart[1] for atom in snapshot.atomlist for snapshot in structure.snaplist]
    zpos = [atom.cart[2] for atom in snapshot.atomlist for snapshot in structure.snaplist]

    structure.box[0][0] = np.amax(xpos) - np.amin(xpos) + 50.0
    structure.box[1][1] = np.amax(ypos) - np.amin(ypos) + 50.0
    structure.box[2][2] = np.amax(zpos) - np.amin(zpos) + 50.0

    structure.make_dircar_matrices()
    structure.cartodir()



    if isinstance(outfilename, list):
        listfilenames = outfilename
    else:
        listfilenames = [outfilename]


    for single_outfilename in list(listfilenames):
        # Read structure
        outfile = open(single_outfilename,'r')
        snapshot_id = 0
        while True:
            line = outfile.readline()
            if not line: # break at EOF
                break

            if 'Mulliken charges and spin densities:' in line:
                dummyline = outfile.readline()
                for atom_index in range(len(atomic_numbers)):
                    structure.snaplist[snapshot_id].atomlist[atom_index].charge = float(outfile.readline().strip().split()[2])
                snapshot_id += 1

    return structure
