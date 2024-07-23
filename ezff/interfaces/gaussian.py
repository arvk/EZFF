"""Interface to Q-Chem, the ab initio quantum chemistry package"""
import numpy as np
import xtal
import cclib
import periodictable as pt


def _general_read(outfilename):
    """
    Wrapper function to read-in a Gaussian16 output/log file. The output file can be from a single-point, optimization, rigid-PES scan or relaxed-PES scan job.

    :param outfilename: Single filename for ``stdout`` from the QChem PES scan job or a list of filenames for ``stdout`` files from partial Gaussian PES scan jobs
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
        data = cclib.io.ccread(single_outfilename)
        atomic_numbers = data.atomnos
        charges = data.atomcharges['mulliken']

        # Look for each following type of job

        try:
            all_configurations = data.scancoords   # Select only optimized coordinates from a scan, if available
            for configuration_id, configuration in enumerate(all_configurations):
                snapshot = structure.create_snapshot(xtal.Snapshot)
                for atom_id, atom_coord in enumerate(configuration):
                    atom = snapshot.create_atom(xtal.Atom)
                    atom.element = str(pt.elements[atomic_numbers[atom_id]])
                    atom.cart = np.array(atom_coord)
            all_energies = data.scanenergies   # Select only optimized energies from a scan, if available
            energies.extend(all_energies)
        except:
            all_configurations = data.atomcoords
            all_energies = data.scfenergies

            if len(all_configurations) == 1: # Is this a single point calculation
                snapshot = structure.create_snapshot(xtal.Snapshot)
                for atom_id, atom_coord in enumerate(all_configurations[0]):
                    atom = snapshot.create_atom(xtal.Atom)
                    atom.element = str(pt.elements[atomic_numbers[atom_id]])
                    atom.cart = np.array(atom_coord)
                energies.extend([all_energies[-1]])
            else:

                if len(all_energies) != len(all_configurations): # If sizes of energy and configuration array don't match, keep only last n_configuration number of energy values
                    all_energies = all_energies[-len(all_configurations):]

                opt_energies = []
                for configuration_id, configuration in enumerate(all_configurations):   # Loop through all configurations looking for converged ones
                    if (data.optstatus[configuration_id] & cclib.parser.data.ccData.OPT_DONE):
                        snapshot = structure.create_snapshot(xtal.Snapshot)
                        for atom_id, atom_coord in enumerate(configuration):
                            atom = snapshot.create_atom(xtal.Atom)
                            atom.element = str(pt.elements[atomic_numbers[atom_id]])
                            atom.cart = np.array(atom_coord)
                        opt_energies.append(all_energies[configuration_id])
                energies.extend(opt_energies)

            if len(structure.snaplist) > 0:
                for atomID, atom in enumerate(structure.snaplist[-1].atomlist):
                    atom.charge = charges[atomID]

    if len(structure.snaplist) > 0:
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

    return (structure, energies)



def read_structure(outfilename):
    structure, _ = _general_read(outfilename)
    return structure



def read_energy(outfilename):
    _, energy = _general_read(outfilename)
    return np.array(energy)



def read_atomic_charges(outfilename):
    structure, _ = _general_read(outfilename)
    return structure
